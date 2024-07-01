/*
 * Copyright (c) Contributors to the Open 3D Engine Project.
 * For complete copyright and license terms please see the LICENSE at the root
 * of this distribution.
 *
 * SPDX-License-Identifier: Apache-2.0 OR MIT
 *
 */

#include "dxc/Support/Global.h"
#include "dxc/Support/Unicode.h"
#include "dxc/Support/WinIncludes.h"
#include "dxc/Support/WinFunctions.h"

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
                                           cl::desc("Output filename"),
                                           cl::value_desc("filename"));

static cl::opt<std::string> OffsetsFilename("f",
                                           cl::desc("Json offsets filename"),
                                           cl::value_desc("filename"));

static cl::opt<int> SentinelValue("sv",
                               cl::desc("Sentinel value to use"));

constexpr uint64_t SentinelMask = 0xffffffffffffff00;
constexpr uint64_t SentinelIdMask = 0x00000000000000ff;

using namespace hlsl::options;

// Writes the offsets into a json file
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

// The DXSC application is used for patching a DXIL blob for use with specialization constants in O3DE.
// It uses a sentinel value to detect the exact offset to each shader constant in the shader bytecode so
// it can be patched at runtime. Based on Godot's approach https://godotengine.org/article/d3d12-adventures-in-shaderland/
#ifdef _WIN32
int __cdecl wmain(int argc, const wchar_t **argv) {
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

    // Patch the LLVM IR.
    for (UINT32 i = 0; i < partCount; ++i) {
      CComPtr<IDxcBlob> pBitcode;
      IFT(pReflection->GetPartContent(i, &pBitcode));

      LPVOID partStart = pBitcode->GetBufferPointer();
      SIZE_T partSize = pBitcode->GetBufferSize();

      UINT32 kind = 0;
      IFT(pReflection->GetPartKind(i, &kind));

      // Find the DXIL container
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

        // Step 1:
        // Make a database of the 'alloca's, based on the volatile stores.
        // %63 = bitcast i32* %59 to i8*
        // call void @llvm.lifetime.start(i64 4, i8* %63)
        // store volatile i32 9, i32* %59, align 4      <---- Locate this
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

        // Step 2:
        // Replace uses of the result of volatile loads by constants, like
        // this:
        //   Original:
        //     %590 = load volatile i32, i32* %61, align 4
        //     %591 = icmp eq i32 %590, 0
        //   Changed:
        //     %586 = icmp eq i32 ???, 0
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

        // Step 3: 
        // Remove any other instructions using the slot the load happened
        // from,
        //   which includes not only volatile stores but also any other
        //   instruction the compiler may have added on its own, such as
        //   'bitcast' and lifetime control calls, as well as any other
        //   instruction that is invalid now due to the previously removed ones.
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

      // Write the container back
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

    // Sign container
    CComPtr<IDxcValidator> pValidator;
    CComPtr<IDxcOperationResult> pResult;
    DxcDllSupport DxilSupport;
    IFT(DxilSupport.InitializeForDll(kDxilLib, "DxcCreateInstance"));
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
    
    // Write the blob
    if (!OutputFilename.empty()) {
      // Write the signed blob to a file
      WriteBlobToFile(pContainerBlob, StringRefWide(OutputFilename), CP_ACP);
    }

    // Write the offsets
    if (!OffsetsFilename.empty())
    {
      WriteOffsetJsonFile((dxilPartOffset + sizeof(hlsl::DxilProgramHeader)) * 8, bitOffsets);
    }

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
    }
    return 1;
  } catch (std::bad_alloc &) {
    llvm::errs() << "failed - out of memory";
    printf("%s failed - out of memory.\n", pStage);
    return 1;
  } catch (...) {
    printf("%s failed - unknown error.\n", pStage);
    return 1;
  }

  printf("Specialized Constants Patching succeeded.");
  return 0;
}
