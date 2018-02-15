/*!
[Later](struct.Later.html) owns the result of a lazy computation which
can be accessed via [reference](struct.Later.html#method.get).  
Works on stable Rust.

# Examples

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
# #[macro_use]
# extern crate later;
# use later::Later;
# fn main() {
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
# }
```
*/

#[cfg(feature = "arrayvec")]
extern crate arrayvec;
#[cfg(feature = "either")]
extern crate either;
#[cfg(feature = "fixedbitset")]
extern crate fixedbitset;
#[cfg(feature = "petgraph")]
extern crate petgraph;

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
    f: Box<Fn() -> T>,
}

/// Unwrap methods for the Later wrapper.
pub trait Failable {
    type Output;

    fn unwrap(self) -> Self::Output;
    fn unwrap_or<I>(self, fallback: Self::Output) -> Self::Output
    where
        I: Into<Self::Output>;
    fn unwrap_or_else<F>(self, f: F) -> Self::Output
    where
        F: FnOnce() -> Self::Output;
    fn unwrap_or_later(self, later: Later<Self::Output>) -> Self::Output;
    fn expect<S: AsRef<str>>(self, msg: S) -> Self::Output;
}

impl<T> Later<T> {
    pub fn new<F>(f: F) -> Later<T>
    where
        F: Fn() -> T + 'static,
    {
        Later {
            cell: LazyCell::new(),
            f: Box::new(f),
        }
    }

    fn with_value(val: T) -> Later<T> {
        let cell = LazyCell::new();
        let _ = cell.fill(val);
        Later {
            cell: cell,
            f: Box::new(|| unreachable!()),
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
    pub fn get_or_default(&self) -> &T
    where
        T: Default,
    {
        if self.cell.filled() {
            self.get()
        } else {
            let _ = self.cell.fill(T::default());
            self.get()
        }
    }

    /// If value has not been computed,
    /// return default otherwise the computation.
    pub fn get_mut_or_default(&mut self) -> &mut T
    where
        T: Default,
    {
        if self.cell.filled() {
            self.cell.borrow_mut().unwrap()
        } else {
            let _ = self.cell.fill(T::default());
            self.cell.borrow_mut().unwrap()
        }
    }

    fn initialize_if_empty(&self) {
        if !self.cell.filled() {
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

    pub fn clone_inner(&self) -> T
    where
        T: Clone,
    {
        self.initialize_if_empty();
        self.get().clone()
    }

    /// Caution: If deferred value has been computed, the value
    /// is cloned to create a new Later<T> object!
    pub fn map<F, U>(self, f: F) -> Later<U>
    where
        F: Fn(T) -> U + 'static,
        T: Clone + 'static, // TODO get rid of Clone
    {
        if self.cell.filled() {
            Later {
                cell: LazyCell::new(),
                f: Box::new(move || f(self.get().clone())),
            }
        } else {
            Later {
                cell: LazyCell::new(),
                f: Box::new(move || f((self.f)())),
            }
        }
    }
}

//-------------------------  Traits  -------------------------//

impl<T> Failable for Later<Option<T>> {
    type Output = T;

    fn unwrap(self) -> T {
        self.into_inner().unwrap()
    }

    fn unwrap_or<I>(self, fallback: T) -> T
    where
        I: Into<T>,
    {
        self.into_inner().unwrap_or_else(|| fallback)
    }

    fn unwrap_or_else<F>(self, f: F) -> Self::Output
    where
        F: FnOnce() -> Self::Output,
    {
        self.into_inner().unwrap_or_else(f)
    }

    fn unwrap_or_later(self, later: Later<T>) -> T {
        self.into_inner().unwrap_or_else(|| later.into_inner())
    }

    fn expect<S: AsRef<str>>(self, msg: S) -> T {
        self.into_inner().expect(msg.as_ref())
    }
}

impl<T, E: std::fmt::Debug> Failable for Later<Result<T, E>> {
    type Output = T;

    fn unwrap(self) -> T {
        self.into_inner().unwrap()
    }

    fn unwrap_or<I>(self, fallback: T) -> T
    where
        I: Into<T>,
    {
        self.into_inner().unwrap_or_else(|_| fallback)
    }

    fn unwrap_or_else<F>(self, f: F) -> Self::Output
    where
        F: FnOnce() -> Self::Output,
    {
        self.into_inner().unwrap_or_else(|_| f())
    }

    fn unwrap_or_later(self, later: Later<T>) -> T {
        self.into_inner().unwrap_or_else(|_| later.into_inner())
    }

    fn expect<S: AsRef<str>>(self, msg: S) -> T {
        self.into_inner().expect(msg.as_ref())
    }
}

impl<T> std::convert::AsRef<Later<T>> for Later<T> {
    fn as_ref(&self) -> &Later<T> {
        self
    }
}

impl<T> std::iter::IntoIterator for Later<T>
where
    T: std::iter::IntoIterator,
{
    type Item = T::Item;
    type IntoIter = <T as std::iter::IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.into_inner().into_iter()
    }
}

impl<T, A> Extend<A> for Later<T>
where
    T: Extend<A>,
{
    fn extend<Y>(&mut self, iter: Y)
    where
        Y: IntoIterator<Item = A>,
    {
        self.get_mut().extend(iter)
    }
}

impl<T> std::fmt::Debug for Later<T>
where
    T: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        if self.cell.filled() {
            write!(f, "Later {{ computed: true, value: {:?} }}", self.get())
        } else {
            write!(f, "Later {{ computed: false, value: unknown }}")
        }
    }
}

impl<T> std::fmt::Display for Later<T>
where
    T: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        if self.cell.filled() {
            write!(f, "(computed: true, value: {})", self.get())
        } else {
            write!(f, "(computed: false, value: unknown)")
        }
    }
}

impl<T> std::cmp::PartialEq for Later<T>
where
    T: std::cmp::PartialEq<T>,
{
    fn eq(&self, rhs: &Later<T>) -> bool {
        self.get().eq(rhs.get())
    }
}

//-------------------------  Into Traits  -------------------------//

impl<T> Into<Option<T>> for Later<Option<T>> {
    fn into(self) -> Option<T> {
        self.into_inner()
    }
}

impl<T, E> Into<Result<T, E>> for Later<Result<T, E>> {
    fn into(self) -> Result<T, E> {
        self.into_inner()
    }
}

impl Into<String> for Later<String> {
    fn into(self) -> String {
        self.into_inner()
    }
}

impl Into<std::path::PathBuf> for Later<std::path::PathBuf> {
    fn into(self) -> std::path::PathBuf {
        self.into_inner()
    }
}

impl<T> Into<Vec<T>> for Later<Vec<T>> {
    fn into(self) -> Vec<T> {
        self.into_inner()
    }
}

#[cfg(feature = "arrayvec")]
impl<T> Into<arrayvec::ArrayVec<T>> for Later<arrayvec::ArrayVec<T>>
where
    T: arrayvec::Array,
{
    fn into(self) -> arrayvec::ArrayVec<T> {
        self.into_inner()
    }
}

#[cfg(feature = "arrayvec")]
impl<T> Into<arrayvec::ArrayString<T>> for Later<arrayvec::ArrayString<T>>
where
    T: arrayvec::Array<Item = u8>,
{
    fn into(self) -> arrayvec::ArrayString<T> {
        self.into_inner()
    }
}

#[cfg(feature = "either")]
impl<L, R> Into<either::Either<L, R>> for Later<either::Either<L, R>> {
    fn into(self) -> either::Either<L, R> {
        self.into_inner()
    }
}

#[cfg(feature = "fixedbitset")]
impl Into<fixedbitset::FixedBitSet> for Later<fixedbitset::FixedBitSet> {
    fn into(self) -> fixedbitset::FixedBitSet {
        self.into_inner()
    }
}

#[cfg(feature = "petgraph")]
impl<N, E, Ty, Ix> Into<petgraph::Graph<N, E, Ty, Ix>> for Later<petgraph::Graph<N, E, Ty, Ix>> {
    fn into(self) -> petgraph::Graph<N, E, Ty, Ix> {
        self.into_inner()
    }
}

#[cfg(feature = "petgraph")]
impl<N, E, Ty, Ix> Into<petgraph::stable_graph::StableGraph<N, E, Ty, Ix>>
    for Later<petgraph::stable_graph::StableGraph<N, E, Ty, Ix>> {
    fn into(self) -> petgraph::stable_graph::StableGraph<N, E, Ty, Ix> {
        self.into_inner()
    }
}

#[cfg(feature = "petgraph")]
impl<N, E, Ty> Into<petgraph::graphmap::GraphMap<N, E, Ty>>
    for Later<petgraph::graphmap::GraphMap<N, E, Ty>> {
    fn into(self) -> petgraph::graphmap::GraphMap<N, E, Ty> {
        self.into_inner()
    }
}

impl<K, V, S> Into<std::collections::HashMap<K, V, S>>
    for Later<std::collections::HashMap<K, V, S>> {
    fn into(self) -> std::collections::HashMap<K, V, S> {
        self.into_inner()
    }
}

impl<T, S> Into<std::collections::HashSet<T, S>> for Later<std::collections::HashSet<T, S>> {
    fn into(self) -> std::collections::HashSet<T, S> {
        self.into_inner()
    }
}

impl<T> Into<std::collections::LinkedList<T>> for Later<std::collections::LinkedList<T>> {
    fn into(self) -> std::collections::LinkedList<T> {
        self.into_inner()
    }
}

impl<T> Into<std::collections::VecDeque<T>> for Later<std::collections::VecDeque<T>> {
    fn into(self) -> std::collections::VecDeque<T> {
        self.into_inner()
    }
}

impl<K, V> Into<std::collections::BTreeMap<K, V>> for Later<std::collections::BTreeMap<K, V>> {
    fn into(self) -> std::collections::BTreeMap<K, V> {
        self.into_inner()
    }
}

impl<T> Into<std::collections::BTreeSet<T>> for Later<std::collections::BTreeSet<T>> {
    fn into(self) -> std::collections::BTreeSet<T> {
        self.into_inner()
    }
}

impl<T> Into<std::collections::BinaryHeap<T>> for Later<std::collections::BinaryHeap<T>> {
    fn into(self) -> std::collections::BinaryHeap<T> {
        self.into_inner()
    }
}

impl Into<i8> for Later<i8> {
    fn into(self) -> i8 {
        self.into_inner()
    }
}

impl Into<i16> for Later<i16> {
    fn into(self) -> i16 {
        self.into_inner()
    }
}

impl Into<i32> for Later<i32> {
    fn into(self) -> i32 {
        self.into_inner()
    }
}

impl Into<i64> for Later<i64> {
    fn into(self) -> i64 {
        self.into_inner()
    }
}

impl Into<u8> for Later<u8> {
    fn into(self) -> u8 {
        self.into_inner()
    }
}

impl Into<u16> for Later<u16> {
    fn into(self) -> u16 {
        self.into_inner()
    }
}

impl Into<u32> for Later<u32> {
    fn into(self) -> u32 {
        self.into_inner()
    }
}

impl Into<u64> for Later<u64> {
    fn into(self) -> u64 {
        self.into_inner()
    }
}

impl Into<isize> for Later<isize> {
    fn into(self) -> isize {
        self.into_inner()
    }
}

impl Into<usize> for Later<usize> {
    fn into(self) -> usize {
        self.into_inner()
    }
}

impl Into<f32> for Later<f32> {
    fn into(self) -> f32 {
        self.into_inner()
    }
}

impl Into<f64> for Later<f64> {
    fn into(self) -> f64 {
        self.into_inner()
    }
}

//-------------------------  Operator Traits  -------------------------//

impl<T> std::ops::Add for Later<T>
where
    T: std::ops::Add<Output = T>,
{
    type Output = Later<T>;

    fn add(self, rhs: Later<T>) -> Later<T> {
        Later::with_value(self.into_inner().add(rhs.into_inner()))
    }
}

impl<T: std::ops::AddAssign<T>> std::ops::AddAssign for Later<T> {
    fn add_assign(&mut self, rhs: Later<T>) {
        self.get_mut().add_assign(rhs.into_inner());
    }
}

impl<T> std::ops::BitAnd for Later<T>
where
    T: std::ops::BitAnd<Output = T>,
{
    type Output = Later<T>;

    fn bitand(self, rhs: Later<T>) -> Later<T> {
        Later::with_value(self.into_inner().bitand(rhs.into_inner()))
    }
}

impl<T: std::ops::BitAndAssign<T>> std::ops::BitAndAssign for Later<T> {
    fn bitand_assign(&mut self, rhs: Later<T>) {
        self.get_mut().bitand_assign(rhs.into_inner());
    }
}

impl<T> std::ops::BitOr for Later<T>
where
    T: std::ops::BitOr<Output = T>,
{
    type Output = Later<T>;

    fn bitor(self, rhs: Later<T>) -> Later<T> {
        Later::with_value(self.into_inner().bitor(rhs.into_inner()))
    }
}

impl<T: std::ops::BitOrAssign<T>> std::ops::BitOrAssign for Later<T> {
    fn bitor_assign(&mut self, rhs: Later<T>) {
        self.get_mut().bitor_assign(rhs.into_inner());
    }
}

impl<T> std::ops::BitXor for Later<T>
where
    T: std::ops::BitXor<Output = T>,
{
    type Output = Later<T>;

    fn bitxor(self, rhs: Later<T>) -> Later<T> {
        Later::with_value(self.into_inner().bitxor(rhs.into_inner()))
    }
}

impl<T: std::ops::BitXorAssign<T>> std::ops::BitXorAssign for Later<T> {
    fn bitxor_assign(&mut self, rhs: Later<T>) {
        self.get_mut().bitxor_assign(rhs.into_inner());
    }
}

impl<T> std::ops::Deref for Later<T> {
    type Target = T;

    fn deref(&self) -> &T {
        self.get()
    }
}

impl<T> std::ops::DerefMut for Later<T> {
    fn deref_mut(&mut self) -> &mut T {
        self.get_mut()
    }
}

impl<T> std::ops::Div for Later<T>
where
    T: std::ops::Div<Output = T>,
{
    type Output = Later<T>;

    fn div(self, rhs: Later<T>) -> Later<T> {
        Later::with_value(self.into_inner().div(rhs.into_inner()))
    }
}

impl<T: std::ops::DivAssign<T>> std::ops::DivAssign for Later<T> {
    fn div_assign(&mut self, rhs: Later<T>) {
        self.get_mut().div_assign(rhs.into_inner());
    }
}

impl<T> std::ops::Index<T> for Later<T>
where
    T: std::ops::Index<T, Output = T>,
{
    type Output = T;

    fn index(&self, v: T) -> &T {
        self.get().index(v)
    }
}

impl<T> std::ops::Mul for Later<T>
where
    T: std::ops::Mul<Output = T>,
{
    type Output = Later<T>;

    fn mul(self, rhs: Later<T>) -> Later<T> {
        Later::with_value(self.into_inner().mul(rhs.into_inner()))
    }
}

impl<T: std::ops::MulAssign<T>> std::ops::MulAssign for Later<T> {
    fn mul_assign(&mut self, rhs: Later<T>) {
        self.get_mut().mul_assign(rhs.into_inner());
    }
}

impl<T> std::ops::Neg for Later<T>
where
    T: std::ops::Neg<Output = T>,
{
    type Output = Later<T>;

    fn neg(self) -> Later<T> {
        Later::with_value(self.into_inner().neg())
    }
}

impl<T> std::ops::Not for Later<T>
where
    T: std::ops::Not<Output = T>,
{
    type Output = Later<T>;

    fn not(self) -> Later<T> {
        Later::with_value(self.into_inner().not())
    }
}

impl<T> std::ops::Rem for Later<T>
where
    T: std::ops::Rem<Output = T>,
{
    type Output = Later<T>;

    fn rem(self, rhs: Later<T>) -> Later<T> {
        Later::with_value(self.into_inner().rem(rhs.into_inner()))
    }
}

impl<T: std::ops::RemAssign<T>> std::ops::RemAssign for Later<T> {
    fn rem_assign(&mut self, rhs: Later<T>) {
        self.get_mut().rem_assign(rhs.into_inner());
    }
}

impl<T> std::ops::Shl<Later<T>> for Later<T>
where
    T: std::ops::Shl<T, Output = T>,
{
    type Output = Later<T>;

    fn shl(self, rhs: Later<T>) -> Later<T> {
        Later::with_value(self.into_inner().shl(rhs.into_inner()))
    }
}

impl<T> std::ops::ShlAssign<Later<T>> for Later<T>
where
    T: std::ops::ShlAssign<T>,
{
    fn shl_assign(&mut self, rhs: Later<T>) {
        self.get_mut().shl_assign(rhs.into_inner());
    }
}

impl<T> std::ops::Shr<Later<T>> for Later<T>
where
    T: std::ops::Shr<T, Output = T>,
{
    type Output = Later<T>;

    fn shr(self, rhs: Later<T>) -> Later<T> {
        Later::with_value(self.into_inner().shr(rhs.into_inner()))
    }
}

impl<T> std::ops::ShrAssign<Later<T>> for Later<T>
where
    T: std::ops::ShrAssign<T>,
{
    fn shr_assign(&mut self, rhs: Later<T>) {
        self.get_mut().shr_assign(rhs.into_inner());
    }
}

impl<T> std::ops::Sub for Later<T>
where
    T: std::ops::Sub<Output = T>,
{
    type Output = Later<T>;

    fn sub(self, rhs: Later<T>) -> Later<T> {
        Later::with_value(self.into_inner().sub(rhs.into_inner()))
    }
}

impl<T: std::ops::SubAssign<T>> std::ops::SubAssign for Later<T> {
    fn sub_assign(&mut self, rhs: Later<T>) {
        self.get_mut().sub_assign(rhs.into_inner());
    }
}

//-------------------------  Binary Number Types  -------------------------//

impl<T> std::fmt::UpperHex for Later<T>
where
    T: std::fmt::UpperHex,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        self.get().fmt(f)
    }
}

impl<T> std::fmt::LowerHex for Later<T>
where
    T: std::fmt::LowerHex,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        self.get().fmt(f)
    }
}

impl<T> std::fmt::Octal for Later<T>
where
    T: std::fmt::Octal,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        self.get().fmt(f)
    }
}

impl<T> std::fmt::Binary for Later<T>
where
    T: std::fmt::Binary,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        self.get().fmt(f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn later() {
        let l = Later::new(|| "abc".to_owned());
        assert!(!l.has_value());
        assert_eq!("abc", l.get());
        assert!(l.has_value());
        assert_eq!("abc", *l);

        assert_eq!(1, later!(1).into_inner());
        println!("{:?}", later!(2));
        println!("{:?}", {
            let l = later!(3);
            l.get();
            l
        });

        assert_eq!(2, later!(Some(2)).unwrap());
    }

    #[test]
    fn unwrap_or_later() {
        let fallack = later!(5);
        assert_eq!(5, later!(None).unwrap_or_later(fallack));
        assert_eq!(3, later!(Some(3)).unwrap());
    }

    #[test]
    fn map() {
        let l1 = later!({
            println!("first");
            100
        });
        assert!(!l1.has_value());
        assert_eq!(100, *l1);
        assert!(l1.has_value());

        let l2 = l1.map(|n| {
            println!("second");
            n.to_string()
        });
        assert!(!l2.has_value());
        assert_eq!("100", *l2);
        assert!(l2.has_value());

        assert_eq!("ab", later!("a".to_owned()).map(|s| s + "b").get());
    }

    #[test]
    fn operators() {
        #[derive(Debug, PartialEq)]
        struct Point {
            x: i32,
            y: i32,
        }

        impl std::ops::Add for Point {
            type Output = Point;

            fn add(self, rhs: Point) -> Point {
                Point {
                    x: self.x + rhs.x,
                    y: self.y + rhs.y,
                }
            }
        }

        let l1 = later!(Point { x: 1, y: 0 });
        let l2 = later!(Point { x: 2, y: 3 });
        assert_eq!(Point { x: 3, y: 3 }, (l1 + l2).into_inner());
        assert_eq!(10, later!(vec![10, 2, 3])[0]);

        assert_eq!(16, *(later!(4) << later!(2)));
        assert_eq!(16, 4 << 2);
    }
}
