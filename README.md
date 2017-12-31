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
    // instead of Later::new the later! macro could be used

    l.has_value();              // false
    let _a: &String = l.get();  // prints: hello from closure
    l.has_value();              // true
    let _b: &String = l.get();  // does not print anything
}
```

```rust
    // the index operator works
    let l = later!(vec![10, 2]);
    assert_eq!(10, l[0]);

    // ... the add-assign operator
    let mut l = later!(95);
    l += later!(5);
    assert_eq!(100, *l);

    // ... and many more
    let l = later!(vec![1, 2]);
    let a = l.into_iter().map(|v| v*10).collect::<Vec<_>>();
    assert_eq!(vec![10, 20], a);
```

## TODO

* Find a way to implement `map` without `Clone`
* Write documentation
* Optional integration with Futures create
* add std::ops::Fn
