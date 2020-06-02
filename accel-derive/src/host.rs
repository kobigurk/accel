use proc_macro2::{Span, TokenStream};
use quote::quote;

/// Split out types from function definition
///
/// - Reference type, e.g. `&i32` will be modified into lifetimed reference `&'arg i32`
///
fn input_types(func: &syn::ItemFn) -> Vec<syn::Type> {
    func.sig
        .inputs
        .iter()
        .map(|arg| match arg {
            syn::FnArg::Typed(ref val) => {
                let mut ty = *val.ty.clone();
                match &mut ty {
                    syn::Type::Reference(re) => {
                        re.lifetime = Some(syn::Lifetime::new("'arg", Span::call_site()))
                    }
                    _ => {}
                }
                ty
            }
            _ => panic!("Unsupported kernel input type sigunature"),
        })
        .collect()
}

fn accel_path() -> String {
    if let Ok(name) = proc_macro_crate::crate_name("accel") {
        // accel exists as an external crate
        return name;
    }

    if std::env::var("CARGO_PKG_NAME").unwrap() == "accel" {
        // doctest in accel
        //
        // "--crate-type bin" should be specified for doctest
        let mut find_flag = false;
        for arg in std::env::args() {
            if arg == "--crate-type" {
                find_flag = true;
            }
            if find_flag {
                if arg == "bin" {
                    return "accel".into();
                }
            }
        }

        // in accel crate
        return "crate".into();
    }
    unreachable!("Cannot determine accel crate name");
}

fn impl_submodule(ptx_str: &str, func: &syn::ItemFn) -> TokenStream {
    let input_types = input_types(func);
    let accel = accel_path();

    let launchable: syn::Path = syn::parse_str(&format!(
        "{}::execution::Launchable{}",
        accel,
        input_types.len()
    ))
    .unwrap();

    let targets: Vec<syn::Ident> = (1..=input_types.len())
        .into_iter()
        .map(|k| syn::Ident::new(&format!("Target{}", k), Span::call_site()))
        .collect();

    let ident = &func.sig.ident;

    let accel = syn::Ident::new(&accel, Span::call_site());
    let kernel_name = quote! { #ident }.to_string();
    quote! {
        /// Auto-generated by accel-derive
        mod #ident {
            pub const PTX_STR: &'static str = #ptx_str;

            pub struct Module(#accel::Module);

            impl Module {
                pub fn new(ctx: &#accel::Context) -> #accel::error::Result<Self> {
                    Ok(Module(#accel::Module::from_str(ctx, PTX_STR)?))
                }
            }

            impl<'arg> #launchable <'arg> for Module {
                #(
                    type #targets = #input_types;
                )*
                fn get_kernel(&self) -> #accel::error::Result<#accel::Kernel> {
                    Ok(self.0.get_kernel(#kernel_name)?)
                }
            }
        }
    }
}

fn caller(func: &syn::ItemFn) -> TokenStream {
    let accel = accel_path();
    let vis = &func.vis;
    let ident = &func.sig.ident;
    let fn_token = &func.sig.fn_token;

    let input_types = input_types(func);

    let args_types: Vec<syn::Ident> = (1..=input_types.len())
        .into_iter()
        .map(|k| syn::Ident::new(&format!("Arg{}", k), Span::call_site()))
        .collect();

    let launchable: syn::Path = syn::parse_str(&format!(
        "{}::execution::Launchable{}",
        accel,
        input_types.len()
    ))
    .unwrap();

    let accel = syn::Ident::new(&accel, Span::call_site());

    quote! {
        #vis #fn_token #ident<'arg, #(#args_types),* >(
            ctx: &#accel::Context,
            grid: impl Into<#accel::Grid>,
            block: impl Into<#accel::Block>,
            args: (#(#args_types,)*)
        ) -> #accel::error::Result<()>
        where
            #(
                #args_types: #accel::execution::DeviceSend<Target = #input_types>
            ),*
        {
            use #launchable;
            let module = #ident::Module::new(ctx)?;
            module.launch(grid, block, args)?;
            Ok(())
        }
    }
}

pub fn func2caller(ptx_str: &str, func: &syn::ItemFn) -> TokenStream {
    let impl_submodule = impl_submodule(ptx_str, func);
    let caller = caller(func);
    quote! {
        #impl_submodule
        #caller
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use std::{
        io::Write,
        process::{Command, Stdio},
    };

    const TEST_KERNEL: &'static str = r#"
    fn kernel_name(arg1: i32, arg2: f64) {}
    "#;

    /// Format TokenStream by rustfmt
    ///
    /// This can test if the input TokenStream is valid in terms of rustfmt.
    fn pretty_print(tt: &impl ToString) -> Result<()> {
        let mut fmt = Command::new("rustfmt")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()?;
        fmt.stdin
            .as_mut()
            .unwrap()
            .write(tt.to_string().as_bytes())?;
        let out = fmt.wait_with_output()?;
        println!("{}", String::from_utf8_lossy(&out.stdout));
        Ok(())
    }

    #[test]
    fn impl_submodule() -> Result<()> {
        let func: syn::ItemFn = syn::parse_str(TEST_KERNEL)?;
        let ts = super::impl_submodule("", &func);
        pretty_print(&ts)?;
        Ok(())
    }

    #[test]
    fn caller() -> Result<()> {
        let func: syn::ItemFn = syn::parse_str(TEST_KERNEL)?;
        let ts = super::caller(&func);
        pretty_print(&ts)?;
        Ok(())
    }
}
