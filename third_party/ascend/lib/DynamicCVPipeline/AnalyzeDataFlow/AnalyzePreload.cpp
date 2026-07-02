/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "ascend/include/DynamicCVPipeline/AnalyzeDataFlow.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "ascend/include/DynamicCVPipeline/Common/BufferCountManager.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"

static constexpr const char *DEBUG_TYPE = "analyze-preload";
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(...)                                                                                                      \
    LLVM_DEBUG({                                                                                                       \
        DBGS();                                                                                                        \
        llvm::dbgs() << __VA_ARGS__;                                                                                   \
        llvm::dbgs() << "\n";                                                                                          \
    })

using namespace mlir;
using namespace triton;
using namespace CVPipeline;

namespace {

// Function names that need preload optimization
static constexpr llvm::StringLiteral preloadFunc[] {
    "",
};

// Check if module contains any function that needs preload optimization
static bool needPreloadOptimization(ModuleOp module)
{
    bool needPreload = false;

    module.walk([&](func::FuncOp funcOp) -> WalkResult {
        if (llvm::is_contained(preloadFunc, funcOp.getSymName())) {
            LDBG("[INFO]: Found function that needs preload optimization: " << funcOp.getSymName());
            needPreload = true;
            return WalkResult::interrupt();
        }
        return WalkResult::advance();
    });

    return needPreload;
}

// Check if scopeOp has CUBE tcore_type
static bool isCubeScope(scope::ScopeOp scopeOp)
{
    auto coreTypeAttr = scopeOp->getAttrOfType<hivm::TCoreTypeAttr>(hivm::TCoreTypeAttr::name);
    if (!coreTypeAttr) {
        return false;
    }
    return coreTypeAttr.getTcoretype() == hivm::TCoreType::CUBE;
}

// Add preload attribute to main_loop forOps inside CUBE scope
static void addPreloadAttrToMainLoop(ModuleOp module)
{
    module.walk([&](scope::ScopeOp scopeOp) {
        // Check if this scope is CUBE
        if (!isCubeScope(scopeOp)) {
            return;
        }
        
        LDBG("[INFO]: Found CUBE scope, adding preload attribute to main_loop forOps inside");
        
        // Walk inside the scope to find main_loop forOps
        scopeOp.walk([&](scf::ForOp forOp) {
            if (forOp->hasAttr(CVPipeline::kMainLoop)) {
                LDBG("[INFO]: Adding preload attribute to main_loop forOp in CUBE scope");
                forOp->setAttr(CVPipeline::kPreload, Builder(module.getContext()).getUnitAttr());
            }
        });
    });
    
    // Set buffer counts for preload optimization
    BufferCountManager::getInstance().setBufferCount(BufferCountManager::DepType::IntraCore, 3);
    BufferCountManager::getInstance().setBufferCount(BufferCountManager::DepType::InterCore, 2);
    LDBG("[INFO]: Set intraBufferCount to 3, interCoreBufferCount to 2");
}

} // namespace

void AnalyzePreloadPass::runOnOperation()
{
    ModuleOp module = getOperation();

    LDBG("Before AnalyzePreload:\n" << module << "\n");

    // Check if any function needs preload optimization
    if (needPreloadOptimization(module)) {
        addPreloadAttrToMainLoop(module);
    }

    LDBG("After AnalyzePreload:\n" << module << "\n");
}

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createAnalyzePreloadPass()
{
    return std::make_unique<AnalyzePreloadPass>();
}

} // namespace triton
} // namespace mlir