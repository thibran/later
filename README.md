# Later

Later owns the result of a lazy computation which can be accessed via reference.  
Works on stable Rust.

Version: 0.1.3

## Installation

Add to `Cargo.toml`:
```toml
[dependencies.later]
git = "https://github.com/thibran/later.git"
tag = "v0.1.3"
```

## Examples

The value `T` of `Later<T>` is evaluated on first access and
stored for later use.

```rust
#[macro_use]
extern crate later;

use later::Later;

fn main() {
    let l: Later<String> = Later::new(|| { 
        println!("hello from closure");
        "foo".to_owned()
    });

    l.has_value();             // false
    let a: &String = l.get();  // prints: hello from closure
    l.has_value();             // true
    let b: &String = l.get();  // does not print anything
}
```

```rust
    // the index operator works
    let l = later!(vec![10, 2]);
    assert_eq!(10, l[0]);

    // ... add-assign
    let mut l = later!(95);
    l += later!(5);
    assert_eq!(100, *l);

    // ... into_iter() and many more
    let a: Vec<u32> = later!(vec![1, 2])
        .into_iter()
        .map(|n| n*10).collect();
```

## TODO

* Find a way to implement `map` without `Clone`
* Write documentation
* Optional integration with the Futures create
* Optional integration with the Serd create
* add std::ops::Fn
* impl traits for [T; n]
