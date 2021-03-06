use proc_macro2::{Span, TokenStream};
use quote::quote;
pub fn generate(item: TokenStream) -> TokenStream {
    let literal: syn::LitInt = syn::parse2(item).unwrap();
    let n: usize = literal.base10_parse().unwrap();
    (0..=n)
        .into_iter()
        .map(|i| {
            let name = syn::Ident::new(&format!("Launchable{}", i), Span::call_site());
            let targets: Vec<syn::Ident> = (1..=i)
                .into_iter()
                .map(|k| syn::Ident::new(&format!("Target{}", k), Span::call_site()))
                .collect();
            let args_value: Vec<syn::Ident> = (1..=i)
                .into_iter()
                .map(|k| syn::Ident::new(&format!("arg{}", k), Span::call_site()))
                .collect();
            let args_types: Vec<syn::Ident> = (1..=i)
                .into_iter()
                .map(|k| syn::Ident::new(&format!("Arg{}", k), Span::call_site()))
                .collect();
            quote! {
                /// Launchable Kernel with N-arguments
                ///
                /// This is auto-generated by `accel_derive::define_launchable!` proc-macro.
                /// See [module level document](index.html) for detail.
                pub trait #name <'arg> {
                    #(
                        type #targets;
                    )*
                    fn get_kernel(&self) -> Result<Kernel>;
                    fn launch<#(#args_types),*>(
                        &self,
                        grid: impl Into<Grid>,
                        block: impl Into<Block>,
                        (#(#args_value,)*): (#(#args_types,)*),
                    ) -> Result<()>
                    where
                        #(
                            #args_types: DeviceSend<Target = Self::#targets>
                        ),*
                    {
                        let grid = grid.into();
                        println!("Launching kernel");
                        println!("Grid: {:?} {:?} {:?}",
                        grid.x,
                        grid.y,
                        grid.z,);
                        let block = block.into();
                        println!("Block: {:?} {:?} {:?}",
                        block.x,
                        block.y,
                        block.z,);
                        let kernel = self.get_kernel()?;
                        let mut args = [#(#args_value.as_kernel_parameter()),*];
                        unsafe {
                            contexted_call!(
                                &kernel,
                                cuLaunchKernel,
                                kernel.func,
                                grid.x,
                                grid.y,
                                grid.z,
                                block.x,
                                block.y,
                                block.z,
                                0,          /* FIXME: no shared memory */
                                null_mut(), /* use default stream */
                                args.as_mut_ptr(),
                                null_mut() /* no extra */
                            )?;
                        }
                        kernel.sync()?;
                        Ok(())
                    }

                    fn launch_async<#(#args_types),*>(
                        &self,
                        grid: impl Into<Grid>,
                        block: impl Into<Block>,
                        (#(#args_value,)*): (#(#args_types,)*),
                    ) -> ::futures::future::BoxFuture<'arg, Result<()>>
                    where
                        #(
                            #args_types: DeviceSend<Target = Self::#targets> + 'arg
                        ),*
                    {
                        let grid = grid.into();
                        let block = block.into();
                        let kernel = self.get_kernel().unwrap();
                        let stream = stream::Stream::new(kernel.get_ref());
                        let mut args = [#(#args_value.as_kernel_parameter()),*];
                        unsafe {
                            contexted_call!(
                                &kernel,
                                cuLaunchKernel,
                                kernel.func,
                                grid.x,
                                grid.y,
                                grid.z,
                                block.x,
                                block.y,
                                block.z,
                                0, /* FIXME: no shared memory */
                                stream.stream,
                                args.as_mut_ptr(),
                                null_mut() /* no extra */
                            )
                        }
                        .expect("Asynchronous kernel launch has been failed");
                        Box::pin(stream.into_future())
                    }
                }
            }
        })
        .collect()
}
