///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// dxa.cpp                                                                   //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// Provides the entry point for the dxa console program.                     //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "dxc/Support/Global.h"
#include "dxc/Support/Unicode.h"
#include "dxc/Support/WinIncludes.h"

#include "dxc/dxcapi.h"
#include "dxc/Support/dxcapi.use.h"
#include "dxc/Support/FileIOHelper.h"
#include "dxc/Support/HLSLOptions.h"
#include "dxc/Support/dxcapi.impl.h"
#include "dxc/DxilContainer/DxilContainer.h"
#include "dxc/DxilRootSignature/DxilRootSignature.h"
#include "dxc/Test/RDATDumper.h"
#include "dxc/Test/D3DReflectionDumper.h"
#include "dxc/DxilContainer/DxilContainerAssembler.h"
#include "dxc/DxilContainer/DxilContainerReader.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support//MSFileSystem.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Bitcode/ReaderWriter.h"

#include <unordered_map>
#include <unordered_set>
#include <map>
#include <filesystem>

using namespace llvm;
using namespace llvm::opt;
using namespace dxc;
using namespace hlsl::options;

static cl::opt<bool> Help("help", cl::desc("Print help"));
static cl::alias Help_h("h", cl::aliasopt(Help));
static cl::alias Help_q("?", cl::aliasopt(Help));

static cl::opt<std::string> InputFilename(cl::Positional,
                                          cl::desc("<<input dxil file>>"));

static cl::opt<std::string> OutputFilename("o",
                                           cl::desc("Override output filename"),
                                           cl::value_desc("filename"));

static cl::opt<std::string> OffsetsFilename("f",
                                           cl::desc("Offset filename"),
                                           cl::value_desc("filename"));

static cl::opt<int> SentinelValue("sv",
                               cl::desc("Sentinel value to use"));

constexpr uint64_t SentinelMask = 0xffffffffffffff00;
constexpr uint64_t SentinelIdMask = 0x00000000000000ff;

using namespace hlsl::options;

std::string BlobToUtf8(_In_ IDxcBlob *pBlob) {
  if (!pBlob)
    return std::string();
  CComPtr<IDxcBlobUtf8> pBlobUtf8;
  if (SUCCEEDED(pBlob->QueryInterface(&pBlobUtf8)))
    return std::string(pBlobUtf8->GetStringPointer(),
                       pBlobUtf8->GetStringLength());
  CComPtr<IDxcBlobEncoding> pBlobEncoding;
  IFT(pBlob->QueryInterface(&pBlobEncoding));
  // if (FAILED(pBlob->QueryInterface(&pBlobEncoding))) {
  //   // Assume it is already UTF-8
  //   return std::string((const char*)pBlob->GetBufferPointer(),
  //                      pBlob->GetBufferSize());
  // }
  BOOL known;
  UINT32 codePage;
  IFT(pBlobEncoding->GetEncoding(&known, &codePage));
  if (!known) {
    throw std::runtime_error("unknown codepage for blob.");
  }
  std::string result;
  if (codePage == DXC_CP_WIDE) {
    const wchar_t *text = (const wchar_t *)pBlob->GetBufferPointer();
    size_t length = pBlob->GetBufferSize() / 2;
    if (length >= 1 && text[length - 1] == L'\0')
      length -= 1; // Exclude null-terminator
    Unicode::WideToUTF8String(text, length, &result);
    return result;
  } else if (codePage == CP_UTF8) {
    const char *text = (const char *)pBlob->GetBufferPointer();
    size_t length = pBlob->GetBufferSize();
    if (length >= 1 && text[length - 1] == '\0')
      length -= 1; // Exclude null-terminator
    result.resize(length);
    memcpy(&result[0], text, length);
    return result;
  } else {
    throw std::runtime_error("Unsupported codepage.");
  }
}

void PatchShader(uint32_t patch_val, uint64_t offset, uint8_t* byteCode) {
  // For VBR encoding to encode the number of bits we expect (32), we need to
  // set the MSB unconditionally. However, signed VBR moves the MSB to the LSB,
  // so setting the MSB to 1 wouldn't help. Therefore, the bit we set to 1 is
  // the one at index 30.
  patch_val <<= 1; // What signed VBR does.

  auto tamper_bits = [](uint8_t *p_start, uint64_t p_bit_offset,
                        uint64_t p_tb_value) -> uint64_t {
    uint64_t original = 0;
    uint32_t curr_input_byte = p_bit_offset / 8;
    uint8_t curr_input_bit = p_bit_offset % 8;
    auto get_curr_input_bit = [&]() -> bool {
      return ((p_start[curr_input_byte] >> curr_input_bit) & 1);
    };
    auto move_to_next_input_bit = [&]() {
      if (curr_input_bit == 7) {
        curr_input_bit = 0;
        curr_input_byte++;
      } else {
        curr_input_bit++;
      }
    };
    auto tamper_input_bit = [&](bool p_new_bit) {
      p_start[curr_input_byte] &= ~((uint8_t)1 << curr_input_bit);
      if (p_new_bit) {
        p_start[curr_input_byte] |= (uint8_t)1 << curr_input_bit;
      }
    };
    uint8_t value_bit_idx = 0;
    for (uint32_t i = 0; i < 5; i++) { // 32 bits take 5 full bytes in VBR.
      for (uint32_t j = 0; j < 7; j++) {
        bool input_bit = get_curr_input_bit();
        original |= (uint64_t)(input_bit ? 1 : 0) << value_bit_idx;
        tamper_input_bit((p_tb_value >> value_bit_idx) & 1);
        move_to_next_input_bit();
        value_bit_idx++;
      }
      if (i < 4)
          p_start[curr_input_byte] |= (uint8_t)1 << curr_input_bit;
      move_to_next_input_bit();
    }
    return original;
  };


#ifdef DEV_ENABLED
    uint64_t orig_patch_val = tamper_bits(bytecode.ptrw(), offset, patch_val);
    // Checking against the value the NIR patch should have set.
    DEV_ASSERT(!p_is_first_patch ||
               ((orig_patch_val >> 1) & GODOT_NIR_SC_SENTINEL_MAGIC_MASK) ==
                   GODOT_NIR_SC_SENTINEL_MAGIC);
    uint64_t readback_patch_val =
        tamper_bits(bytecode.ptrw(), offset, patch_val);
    DEV_ASSERT(readback_patch_val == patch_val);
#else
    tamper_bits(byteCode, offset, patch_val);
#endif
}

void WriteOffsetJsonFile(
    uint64_t dxilPartOffset, const std::map<uint32_t, uint64_t>& offsets) {
  std::stringstream stream;
  stream << "{";
  for (const auto &entry : offsets) {
    stream << "\n\t\"" << entry.first << "\": " << (dxilPartOffset + entry.second) << ",";
  }
  if (!offsets.empty())
  {
      // Remove last ","
    stream.seekp(-1, stream.cur);
  }
  stream << "\n}";
  auto jsonString = stream.str();

  StringRefWide pFileName = StringRefWide(OffsetsFilename);
  CHandle file(CreateFileW(pFileName, GENERIC_WRITE, FILE_SHARE_READ, nullptr,
                           CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr));
  if (file == INVALID_HANDLE_VALUE) {
    IFT_Data(HRESULT_FROM_WIN32(GetLastError()), pFileName);
  }

    DWORD written;

  if (FALSE == WriteFile(file, jsonString.data(), (DWORD)jsonString.size(), &written, nullptr)) {
    IFT_Data(HRESULT_FROM_WIN32(GetLastError()), pFileName);
  }
}

#ifdef _WIN32
int __cdecl main(int argc, const char **argv) {
#else
int main(int argc, const char **argv) {
#endif
  if (llvm::sys::fs::SetupPerThreadFileSystem())
    return 1;
  llvm::sys::fs::AutoCleanupPerThreadFileSystem auto_cleanup_fs;
  if (FAILED(DxcInitThreadMalloc())) return 1;
  DxcSetThreadMallocToDefault();

  const char *pStage = "Operation";
  try {
    llvm::sys::fs::MSFileSystem *msfPtr;
    IFT(CreateMSFileSystemForDisk(&msfPtr));
    std::unique_ptr<::llvm::sys::fs::MSFileSystem> msf(msfPtr);

    ::llvm::sys::fs::AutoPerThreadSystem pts(msf.get());
    IFTLLVM(pts.error_code());

    pStage = "Argument processing";

    // Parse command line options.
    MainArgs argStrings(argc, argv, 0);
    cl::ParseCommandLineOptions(argc, argStrings.getArrayRef().data(),
                                "dxsc patching\n");

    if (InputFilename == "" || !SentinelValue || Help) {
      cl::PrintHelpMessage();
      return 2;
    }

    DxcDllSupport dxcSupport;
    dxc::EnsureEnabled(dxcSupport);
    
    CComPtr<IDxcBlobEncoding> pSource;
    ReadFileIntoBlob(dxcSupport, StringRefWide(InputFilename), &pSource);

    CComPtr<IDxcContainerReflection> pReflection;
    IFT(dxcSupport.CreateInstance(CLSID_DxcContainerReflection,
                                    &pReflection));
    IFT(pReflection->Load(pSource));

    std::unique_ptr<hlsl::DxilContainerWriter> pContainerWriter(
        hlsl::NewDxilContainerWriter(true));
    CComPtr<hlsl::AbstractMemoryStream> patchedDxilPartStream;

    std::map<uint32_t, uint64_t> bitOffsets;
    uint64_t dxilPartOffset = 0;
    UINT32 partCount = 0;
    IFT(pReflection->GetPartCount(&partCount));

    for (UINT32 i = 0; i < partCount; ++i) {
      CComPtr<IDxcBlob> pBitcode;
      IFT(pReflection->GetPartContent(i, &pBitcode));

      LPVOID partStart = pBitcode->GetBufferPointer();
      SIZE_T partSize = pBitcode->GetBufferSize();

      UINT32 kind = 0;
      IFT(pReflection->GetPartKind(i, &kind));

      if (kind == hlsl::DFCC_DXIL) {
        const hlsl::DxilProgramHeader *dxilHeader =
            (const hlsl::DxilProgramHeader *)partStart;

        const char *bitcode_ptr = hlsl::GetDxilBitcodeData(dxilHeader);
        unsigned bitcode_size = hlsl::GetDxilBitcodeSize(dxilHeader);

        llvm::StringRef bitcode_buffer(bitcode_ptr, bitcode_size);
        llvm::LLVMContext llvm_context;
        llvm::ErrorOr<std::unique_ptr<llvm::Module>> bitcode_parsed =
            llvm::parseBitcodeFile(llvm::MemoryBufferRef(bitcode_buffer, ""),
                                   llvm_context);
        if (std::error_code ec = bitcode_parsed.getError()) {
          throw hlsl::Exception(DXC_E_MALFORMED_CONTAINER);
        }

        llvm::Module &llvm_module = *bitcode_parsed->get();
        std::unordered_map<llvm::AllocaInst *, uint32_t> allocSCIndices;

        // Find the specialization constants that are volatile variables
        for (llvm::Function &function : llvm_module.getFunctionList()) {
          if (function.isDeclaration()) {
            continue;
          }
          for (llvm::BasicBlock &block : function) {
            for (llvm::Instruction &instruction : block) {
              if (llvm::isa<llvm::StoreInst>(instruction) &&
                  llvm::cast<llvm::StoreInst>(instruction).isVolatile()) {
                llvm::StoreInst &store =
                    llvm::cast<llvm::StoreInst>(instruction);
                uint32_t scIndex =
                    llvm::dyn_cast<llvm::ConstantInt>(store.getValueOperand())
                        ->getZExtValue();
                llvm::AllocaInst *allocaInst =
                    llvm::dyn_cast<llvm::AllocaInst>(store.getPointerOperand());
                allocSCIndices[allocaInst] = scIndex;
              }
            }
          }
        }

        std::vector<llvm::Instruction *> instructionsToRemove;
        std::unordered_set<llvm::Instruction *> instructionsToRemoveUnique;
        std::unordered_map<llvm::LoadInst *, uint32_t> loadsToReplaceToSCindex;

        // Collect the load operations for the specialization constants
        for (auto &E : allocSCIndices) {
          llvm::AllocaInst *alloca_inst = E.first;
          uint32_t scIndex = E.second;

          if (instructionsToRemoveUnique.find(alloca_inst) ==
              instructionsToRemoveUnique.end()) {
            instructionsToRemove.push_back(alloca_inst);
            instructionsToRemoveUnique.insert(alloca_inst);

            for (llvm::User *U : alloca_inst->users()) {
              if (llvm::isa<llvm::LoadInst>(U) &&
                  llvm::cast<llvm::LoadInst>(U)->isVolatile()) {
                llvm::LoadInst *LI = llvm::cast<llvm::LoadInst>(U);
                loadsToReplaceToSCindex[LI] = scIndex;
              }
            }
          }
        }

        // Replace all loads with the sentinel value + the specialization constant index
        for (auto &E : loadsToReplaceToSCindex) {
          llvm::LoadInst *I = E.first;
          uint32_t scIndex = E.second;
          llvm::Constant *scConstant = llvm::ConstantInt::get(
              llvm::Type::getInt32Ty(llvm_context), SentinelValue + scIndex);
          I->replaceAllUsesWith(scConstant);
        }

        // Collect the rest of the instructions to remove
        uint32_t k = 0;
        while (k < instructionsToRemove.size()) {
          for (llvm::User *U : instructionsToRemove[k]->users()) {
            llvm::Instruction *I = llvm::cast<llvm::Instruction>(U);
            if (instructionsToRemoveUnique.find(I) ==
                instructionsToRemoveUnique.end())
            {
              instructionsToRemove.push_back(I);
              instructionsToRemoveUnique.insert(I);
            }
          }
          k++;
        }

        // Remove all unnecesary operations
        for (llvm::Instruction* instruction : instructionsToRemove) {

          instruction->removeFromParent();
          instruction->dropAllReferences();
        }
        for (llvm::Instruction *instruction : instructionsToRemove) {
          delete instruction;
        }

        // Now rebuild the DXIL part.
        {
          IFR(hlsl::CreateMemoryStream(DxcGetThreadMallocNoRef(),
                                       &patchedDxilPartStream));

          ULONG headerSize = 0;
          IFR(patchedDxilPartStream->Write(
              dxilHeader, sizeof(hlsl::DxilProgramHeader),
                             &headerSize));

          ULONG bitcode_start = patchedDxilPartStream->GetPtrSize();
          {
            raw_stream_ostream ostream(patchedDxilPartStream);
            llvm::WriteBitcodeToFile(
                &llvm_module, ostream, false,
                                     [&](uint64_t value, uint64_t offset)
                {
                    // Check if it's the sentinel value we replaced
                    if ((value & SentinelMask) == (uint64_t)SentinelValue)
                    {
                      // Extract the specialization index id and save the offset for patching
                      uint32_t scIndex = (uint32_t)(value & SentinelIdMask);
                      bitOffsets[scIndex] = offset;
                    }
                });
          }
          ULONG bitcodeEnd = patchedDxilPartStream->GetPtrSize();

          partStart = patchedDxilPartStream->GetPtr();
          partSize = patchedDxilPartStream->GetPtrSize();

          ((hlsl::DxilProgramHeader *)partStart)->SizeInUint32 =
              bitcodeEnd / sizeof(uint32_t);
          ((hlsl::DxilProgramHeader *)partStart)->BitcodeHeader.BitcodeSize =
              bitcodeEnd - bitcode_start;
        }
      }

      // Write part
      pContainerWriter->AddPart(
          kind, partSize,
          [=, &dxilPartOffset](hlsl::AbstractMemoryStream *s) {
            if (kind == hlsl::DFCC_DXIL) {
              dxilPartOffset = s->GetPosition();
            }
            ULONG written = 0;
            s->Write(partStart, partSize, &written);
          });
    }

    // Serialize the container
    CComPtr<IDxcUtils> pUtils;
    IFT(DxcCreateInstance(CLSID_DxcUtils, IID_PPV_ARGS(&pUtils)));

    CComPtr<IDxcBlobEncoding> pContainerBlob;
    UINT32 OutputSize = pContainerWriter->size();
    CComHeapPtr<void> pOutput;
    IFTBOOL(pOutput.AllocateBytes(OutputSize), E_OUTOFMEMORY);
    CComPtr<hlsl::AbstractMemoryStream> pOutputStream;
    IFT(CreateFixedSizeMemoryStream((LPBYTE)pOutput.m_pData, OutputSize,
                                    &pOutputStream));
    pContainerWriter->write(pOutputStream);
    IFR(pUtils->CreateBlob((LPBYTE)pOutput.m_pData, OutputSize, DXC_CP_ACP,
                           &pContainerBlob));

    // Patch
    /*uint64_t abs_sc_sentinel_bit_offset = dxilPartOffset * 8 +
                                          sizeof(hlsl::DxilProgramHeader) * 8 +
                                          bitOffsets[3];
    PatchShader(1, abs_sc_sentinel_bit_offset,
                (uint8_t *)pContainerBlob->GetBufferPointer());

    abs_sc_sentinel_bit_offset = dxilPartOffset * 8 +
                                 sizeof(hlsl::DxilProgramHeader) * 8 +
                                 bitOffsets[1];
    PatchShader(0, abs_sc_sentinel_bit_offset,
                (uint8_t *)pContainerBlob->GetBufferPointer());

    abs_sc_sentinel_bit_offset = dxilPartOffset * 8 +
                                 sizeof(hlsl::DxilProgramHeader) * 8 +
                                 bitOffsets[2];
    PatchShader(0, abs_sc_sentinel_bit_offset,
            (uint8_t *)pContainerBlob->GetBufferPointer());*/

    // Sign container
    CComPtr<IDxcValidator> pValidator;
    CComPtr<IDxcOperationResult> pResult;
    DxcDllSupport DxilSupport;
    HRESULT __hr = DxilSupport.InitializeForDll(kDxilLib, "DxcCreateInstance");
    if (DXC_FAILED(__hr))
      throw ::hlsl::Exception(__hr);

    IFT(DxilSupport.CreateInstance(CLSID_DxcValidator, &pValidator));
    IFT(pValidator->Validate(pContainerBlob, DxcValidatorFlags_InPlaceEdit,
                             &pResult));

    HRESULT status;
    IFT(pResult->GetStatus(&status));

    if (FAILED(status)) {
      // Signing failed.
      CComPtr<IDxcBlobEncoding> text;
      IFT(pResult->GetErrorBuffer(&text));
      const char *pStart = (const char *)text->GetBufferPointer();
      std::string msg(pStart);
      IFTMSG(status, msg);

    } 
    
    if (!OutputFilename.empty()) {
      // Write the signed blob to a file
      WriteBlobToFile(pContainerBlob, StringRefWide(OutputFilename), CP_ACP);
      printf("Specialized Constants Patching succeeded.");
    }

    if (!OffsetsFilename.empty())
    {
      WriteOffsetJsonFile((dxilPartOffset + sizeof(hlsl::DxilProgramHeader)) * 8, bitOffsets);
    }

    //CComPtr<IDxcCompiler> pCompiler;
    //IFT(dxcSupport.CreateInstance(CLSID_DxcCompiler, &pCompiler));
    //CComPtr<IDxcBlobEncoding> pDisassembleBlob;
    //IFR(pCompiler->Disassemble(pContainerBlob, &pDisassembleBlob));
    //std::string disassembleString(BlobToUtf8(pDisassembleBlob));
    //printf("%s", disassembleString.c_str());

  } catch (const ::hlsl::Exception &hlslException) {
    try {
      const char *msg = hlslException.what();
      Unicode::acp_char printBuffer[128]; // printBuffer is safe to treat as
                                          // UTF-8 because we use ASCII only errors
                                          // only
      if (msg == nullptr || *msg == '\0') {
        sprintf_s(printBuffer, _countof(printBuffer),
                  "Assembly failed - error code 0x%08x.", hlslException.hr);
        msg = printBuffer;
      }
      llvm::errs() << msg;
      printf("%s\n", msg);
    } catch (...) {
      printf("%s failed - unable to retrieve error message.\n", pStage);
      llvm::errs() << "unable to retrieve error message";
    }
    llvm::errs().flush();
    return 1;
  } catch (std::bad_alloc &) {
    llvm::errs() << "failed - out of memory";
    printf("%s failed - out of memory.\n", pStage);
    llvm::errs().flush();
    return 1;
  } catch (...) {
    llvm::errs() << "failed - unknown error";
    printf("%s failed - unknown error.\n", pStage);
    llvm::errs().flush();
    return 1;
  }

  return 0;
}
