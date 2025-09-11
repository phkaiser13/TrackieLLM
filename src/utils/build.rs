// src/utils/build.rs

use std::env;
use std::path::PathBuf;

fn main() {
    // 1. Tell Cargo to re-run this script if the C header file changes.
    // This ensures that our generated bindings are always up-to-date.
    println!("cargo:rerun-if-changed=tk_error_handling.h");

    // 2. Configure bindgen to generate the bindings.
    let bindings = bindgen::Builder::default()
        // The input header we want to generate bindings for.
        .header("tk_error_handling.h")
        // Only generate bindings for the specific enum we need. This avoids
        // pulling in a lot of unnecessary C standard library types.
        .allowlist_type("tk_error_code_t")
        // Tell bindgen to use standard Rust enums for C enums.
        .rustified_enum("tk_error_code_t")
        // Invalidate the build whenever the wrapped header changes.
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        // Finish the builder and generate the bindings.
        .generate()
        // Unwrap the Result, panicking on error. If bindgen fails, we want
        // the build to fail with a clear message.
        .expect("Unable to generate bindings for tk_error_handling.h");

    // 3. Write the bindings to the $OUT_DIR/bindings.rs file.
    // The `OUT_DIR` is a directory that Cargo provides for build scripts to
    // place generated code.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
