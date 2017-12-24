# Later

Later owns the result of a lazy computation which can be accessed via reference.

Version: 0.1.0

## Installation

Add to `Cargo.toml`:
```toml
[dependencies.later]
git = "https://github.com/thibran/later.git"
tag = "v0.1.0"
```

## Example

The value `T` of a `Later<T>` is evaluated on first container access
and stored for later use.

```rust
#[macro_use]
extern crate later;

use later::Later;

fn main() {
    let l: Later<String> = Later::new(|| { 
        println!("hello from closure");
        "foo".to_owned()
    });
    // instead of Later::new the defer! macro could be used

    l.has_value();              // false
    let _a: &String = l.get();  // prints: hello from closure
    l.has_value();              // true
    let _b: &String = l.get();  // does not print anything
}
```

## TODO

* Find a way to implement `map` without `Clone`
* Write documentation
* Optional integration with Futures create
