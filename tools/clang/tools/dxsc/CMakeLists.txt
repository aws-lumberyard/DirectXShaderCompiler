# Copyright (C) Microsoft Corporation. All rights reserved.
# This file is distributed under the University of Illinois Open Source License. See LICENSE.TXT for details.
# Builds dxsc.exe

set( LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  DXIL
  DxilContainer
  DxilRootSignature
  HLSL
  dxcsupport
  Option     # option library
  MSSupport  # for CreateMSFileSystemForDisk
  )

add_clang_executable(dxsc
  dxsc.cpp
  )

target_link_libraries(dxsc
  dxcompiler
  )

set_target_properties(dxsc PROPERTIES VERSION ${CLANG_EXECUTABLE_VERSION})

add_dependencies(dxsc dxcompiler)

if(UNIX)
  set(CLANGXX_LINK_OR_COPY create_symlink)
# Create a relative symlink
  set(dxsc_binary "dxsc${CMAKE_EXECUTABLE_SUFFIX}")
else()
  set(CLANGXX_LINK_OR_COPY copy)
  set(dxsc_binary "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CMAKE_CFG_INTDIR}/dxsc${CMAKE_EXECUTABLE_SUFFIX}")
endif()

install(TARGETS dxsc
  RUNTIME DESTINATION bin)

