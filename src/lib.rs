//! [Later](struct.Later.html) owns the result of a lazy computation which
//! can be accessed via [reference](struct.Later.html#method.get).
//!
//! # Example
//! 
//! The value `T` of a `Later<T>` is evaluated on first container access
//! and stored for later use.
//!
//! ```rust
//! #[macro_use]
//! extern crate later;
//!
//! use later::Later;
//!
//! fn main() {
//!     let l: Later<String> = Later::new(|| { 
//!         println!("hello from closure");
//!         "foo".to_owned()
//!     });
//!     // instead of Later::new the later! macro could be used
//!
//!     l.has_value();              // false
//!     let _a: &String = l.get();  // prints: hello from closure
//!     l.has_value();              // true
//!     let _b: &String = l.get();  // does not print anything
//! }
//! ```

extern crate lazycell;

use lazycell::LazyCell;

#[macro_export]
macro_rules! later {
    ($e:expr) => {
        $crate::Later::new(move || $e)
    }
}


pub struct Later<T> {
    cell: LazyCell<T>,
    f:    Box<Fn() -> T>,
}

impl<T> Later<T> {
    pub fn new<F>(f: F) -> Later<T>
        where F: Fn() -> T + 'static
    {
        Later {
            cell: LazyCell::new(),
            f: Box::new(f),
        }
    }

    pub fn get(&self) -> &T {
        self.cell.borrow_with(|| (self.f)())
    }

    pub fn get_mut(&mut self) -> &mut T {
        self.initialize_if_empty();
        self.cell.borrow_mut().unwrap()
    }

    /// If value has not been computed,
    /// return default otherwise the computation. 
    pub fn get_or_default(&self) -> &T where T: Default {
        match self.cell.filled() {
            true => self.get(),
            _    => {
                let _ = self.cell.fill(T::default());
                self.get()
            },
        }
    }

    /// If value has not been computed,
    /// return default otherwise the computation. 
    pub fn get_mut_or_default(&mut self) -> &mut T where T: Default {
        match self.cell.filled() {
            true => self.cell.borrow_mut().unwrap(),
            _    => {
                let _ = self.cell.fill(T::default());
                self.cell.borrow_mut().unwrap()
            },
        }
    }

    fn initialize_if_empty(&self) {
        if ! self.cell.filled() {
            self.get();
        }
    }

    pub fn has_value(&self) -> bool {
        self.cell.filled()
    }

    pub fn into_inner(self) -> T {
        self.initialize_if_empty();
        self.cell.into_inner().unwrap() // is save, value must be present
    }

    pub fn clone_inner(&self) -> T where T: Clone {
        self.initialize_if_empty();
        self.get().clone()
    }

    /// Caution: If deferred value has been computed, map needs
    /// to clone it to create a new Later<T> object!
    pub fn map<F, U>(self, f: F) -> Later<U>
        where F: Fn(T) -> U + 'static,
              T: Clone      + 'static  // TODO get rid of Clone
    {
        match self.cell.filled() {
            true => Later {
                cell: LazyCell::new(),
                f: Box::new(move || f(self.get().clone())),
            },
            _ => Later {
                cell: LazyCell::new(),
                f: Box::new(move || f((self.f)())),
            },
        }
    }
}


//-------------------------  Traits  -------------------------//

impl<T> std::convert::AsRef<Later<T>> for Later<T> {
    #[inline(always)]
    fn as_ref(&self) -> &Later<T> {
        &self
    }
}

// keep Deref?
impl<T> std::ops::Deref for Later<T> {
    type Target = T;

    #[inline(always)]
    fn deref(&self) -> &T {
        self.get()
    }
}

impl<T> std::iter::IntoIterator for Later<T> 
    where T: std::iter::IntoIterator
{
    type Item = T::Item;
    type IntoIter = <T as std::iter::IntoIterator>::IntoIter;

    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        self.into_inner().into_iter()
    }
}

impl<T> std::fmt::Debug for Later<T>
    where T: std::fmt::Debug
{
    #[inline(always)]
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self.cell.filled() {
            true => write!(f, "Later {{ computed: true, value: {:?} }}", self.get()),
            _    => write!(f, "Later {{ computed: false, value: unknown }}"),
        }
    }
}

impl<T> std::fmt::Display for Later<T>
    where T: std::fmt::Display
{
    #[inline(always)]
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self.cell.filled() {
            true => write!(f, "(computed: true, value: {})", self.get()),
            _    => write!(f, "(computed: false, value: unknown)"),
        }
    }
}


//-------------------------  Into Traits  -------------------------//

impl<T> Into<Option<T>> for Later<Option<T>> {
    #[inline(always)]
    fn into(self) -> Option<T> {
        self.into_inner()
    }
}

impl<T, E> Into<Result<T, E>> for Later<Result<T, E>> {
    #[inline(always)]
    fn into(self) -> Result<T, E> {
        self.into_inner()
    }
}

impl Into<String> for Later<String> {
    #[inline(always)]
    fn into(self) -> String {
        self.into_inner()
    }
}

impl Into<std::path::PathBuf> for Later<std::path::PathBuf> {
    #[inline(always)]
    fn into(self) -> std::path::PathBuf {
        self.into_inner()
    }
}

impl<T> Into<Vec<T>> for Later<Vec<T>> {
    #[inline(always)]
    fn into(self) -> Vec<T> {
        self.into_inner()
    }
}

impl<K, V, S> Into<std::collections::HashMap<K, V, S>> for
    Later<std::collections::HashMap<K, V, S>>
{
    #[inline(always)]
    fn into(self) -> std::collections::HashMap<K, V, S> {
        self.into_inner()
    }
}

impl<T, S> Into<std::collections::HashSet<T, S>> for
    Later<std::collections::HashSet<T, S>>
{
    #[inline(always)]
    fn into(self) -> std::collections::HashSet<T, S> {
        self.into_inner()
    }
}

impl<T> Into<std::collections::LinkedList<T>> for
    Later<std::collections::LinkedList<T>>
{
    #[inline(always)]
    fn into(self) -> std::collections::LinkedList<T> {
        self.into_inner()
    }
}

impl<T> Into<std::collections::VecDeque<T>> for
    Later<std::collections::VecDeque<T>>
{
    #[inline(always)]
    fn into(self) -> std::collections::VecDeque<T> {
        self.into_inner()
    }
}

impl<K, V> Into<std::collections::BTreeMap<K, V>> for
    Later<std::collections::BTreeMap<K, V>>
{
    #[inline(always)]
    fn into(self) -> std::collections::BTreeMap<K, V> {
        self.into_inner()
    }
}

impl<T> Into<std::collections::BTreeSet<T>> for
    Later<std::collections::BTreeSet<T>>
{
    #[inline(always)]
    fn into(self) -> std::collections::BTreeSet<T> {
        self.into_inner()
    }
}

impl<T> Into<std::collections::BinaryHeap<T>> for
    Later<std::collections::BinaryHeap<T>>
{
    #[inline(always)]
    fn into(self) -> std::collections::BinaryHeap<T> {
        self.into_inner()
    }
}

impl Into<i8> for Later<i8> {
    #[inline(always)]
    fn into(self) -> i8 {
        self.into_inner()
    }
}

impl Into<i16> for Later<i16> {
    #[inline(always)]
    fn into(self) -> i16 {
        self.into_inner()
    }
}

impl Into<i32> for Later<i32> {
    #[inline(always)]
    fn into(self) -> i32 {
        self.into_inner()
    }
}

impl Into<i64> for Later<i64> {
    #[inline(always)]
    fn into(self) -> i64 {
        self.into_inner()
    }
}

impl Into<u8> for Later<u8> {
    #[inline(always)]
    fn into(self) -> u8 {
        self.into_inner()
    }
}

impl Into<u16> for Later<u16> {
    #[inline(always)]
    fn into(self) -> u16 {
        self.into_inner()
    }
}

impl Into<u32> for Later<u32> {
    #[inline(always)]
    fn into(self) -> u32 {
        self.into_inner()
    }
}

impl Into<u64> for Later<u64> {
    #[inline(always)]
    fn into(self) -> u64 {
        self.into_inner()
    }
}

impl Into<isize> for Later<isize> {
    #[inline(always)]
    fn into(self) -> isize {
        self.into_inner()
    }
}

impl Into<usize> for Later<usize> {
    #[inline(always)]
    fn into(self) -> usize {
        self.into_inner()
    }
}

impl Into<f32> for Later<f32> {
    #[inline(always)]
    fn into(self) -> f32 {
        self.into_inner()
    }
}

impl Into<f64> for Later<f64> {
    #[inline(always)]
    fn into(self) -> f64 {
        self.into_inner()
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_later() {
        let d = Later::new(|| "abc".to_owned());
        assert!(! d.has_value());
        assert_eq!("abc", d.get());
        assert!(d.has_value());
        assert_eq!("abc", *d);

        assert_eq!(1, later!(1).into_inner());
        println!("{:?}", later!(2));
        println!("{:?}", {let d = later!(3); d.get(); d}); 
    }

    #[test]
    fn test_map() {
        let d1 = later!({
            println!("first");
            100
        });
        assert!(! d1.has_value());
        assert_eq!(100, *d1);
        assert!( d1.has_value());

        let d2 = d1.map(|n| {
            println!("second");
            n.to_string()
        });
        assert!(! d2.has_value());
        assert_eq!("100", *d2);
        assert!(d2.has_value());

        assert_eq!("ab", later!("a".to_owned()).map(|s| s+"b").get());
    }
}
