//! CUDA Module (i.e. loaded PTX or cubin)

use crate::{contexted_call, contexted_new, device::*, error::*, *};
use cuda::*;
use std::ffi::*;

/// CUDA Kernel function
#[derive(Debug)]
pub struct Kernel<'module> {
    pub(crate) func: CUfunction,
    module: &'module Module,
}

impl Contexted for Kernel<'_> {
    fn sync(&self) -> Result<()> {
        self.module.context.sync()
    }

    fn version(&self) -> Result<u32> {
        self.module.context.version()
    }

    fn guard(&self) -> Result<ContextGuard> {
        self.module.context.guard()
    }

    fn get_ref(&self) -> ContextRef {
        self.module.get_ref()
    }
}

/// OOP-like wrapper of `cuModule*` APIs
#[derive(Debug, Contexted)]
pub struct Module {
    module: CUmodule,
    context: Context,
}

impl Drop for Module {
    fn drop(&mut self) {
        if let Err(e) = unsafe { contexted_call!(&self.context, cuModuleUnload, self.module) } {
            log::error!("Failed to unload module: {:?}", e);
        }
    }
}

impl Module {
    /// integrated loader of Instruction
    pub fn load(context: &Context, data: &Instruction) -> Result<Self> {
        match *data {
            Instruction::PTX(ref ptx) => {
                let module =
                    unsafe { contexted_new!(context, cuModuleLoadData, ptx.as_ptr() as *const _)? };
                Ok(Module {
                    module,
                    context: context.clone(),
                })
            }
            Instruction::Cubin(ref bin) => {
                let module =
                    unsafe { contexted_new!(context, cuModuleLoadData, bin.as_ptr() as *const _)? };
                Ok(Module {
                    module,
                    context: context.clone(),
                })
            }
            Instruction::PTXFile(ref path) | Instruction::CubinFile(ref path) => {
                let filename = CString::new(path.to_str().unwrap()).expect("Invalid Path");
                let module = unsafe { contexted_new!(context, cuModuleLoad, filename.as_ptr())? };
                Ok(Module {
                    module,
                    context: context.clone(),
                })
            }
        }
    }

    pub fn from_str(context: &Context, ptx: &str) -> Result<Self> {
        let data = Instruction::ptx(ptx);
        Self::load(context, &data)
    }

    /// Wrapper of `cuModuleGetFunction`
    pub fn get_kernel(&self, name: &str) -> Result<Kernel> {
        let name = CString::new(name).expect("Invalid Kernel name");
        let func =
            unsafe { contexted_new!(self, cuModuleGetFunction, self.module, name.as_ptr()) }?;
        Ok(Kernel { func, module: self })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_do_nothing() -> Result<()> {
        // generated by do_nothing example in accel-derive
        let ptx = r#"
        .version 3.2
        .target sm_30
        .address_size 64
        .visible .entry do_nothing()
        {
          ret;
        }
        "#;
        let device = Device::nth(0)?;
        let ctx = device.create_context();
        let _mod = Module::from_str(&ctx, ptx)?;
        Ok(())
    }
}
