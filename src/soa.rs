use collections::vec;
use core::cmp::Ordering;
use core::default::Default;
use core::fmt::{Debug, Formatter, Result};
use core::hash::{self, Hash};
use core::iter::{self, repeat};
use core::mem;
use core::num::Int;
use core::ptr;
use core::slice;
use std::slice::SliceExt;
use unadorned::{self, Unadorned, Extent};

macro_rules! first {
    ($x:expr, $($xs:expr),*) => { $x }
}

macro_rules! anyrest {
    ($x:expr, $($xs:expr),*) => { $($xs)||* }
}

macro_rules! gen_soa {
    ($soa:ident, $soazip:ident, $soazipm:ident | $($ty:ident),+ | $($nm:ident),+ | $($nmu:ident),+) => {
        /// A growable struct-of-N-arrays type, with heap allocated contents.
        ///
        /// This structure is analogous to a `Vec<(A, B, ...)>`, but
        /// instead of laying out the tuples sequentially in memory,
        /// each row gets its own allocation. For example, an
        /// `Soa2<f32, i64>` will contain two inner arrays: one of
        /// `f32`s, and one of `i64`s.

        #[unsafe_no_drop_flag]
        pub struct $soa<$($ty),+> {
            $($nm: Unadorned<$ty>),+,
            e: Extent,
        }

        pub struct $soazip<'a, $($ty:'a),+> {
            parent: &'a $soa<$($ty),+>,
            i: usize
        }

        impl<'a, $($ty),+> Iterator for $soazip<'a, $($ty),+> {
            type Item = ($(&'a $ty),+);
            #[inline]
            fn next(&mut self) -> Option<($(&'a $ty),+)> {
                let i = self.i;
                if i == self.parent.len() { return None }
                self.i += 1;
                let ($($nm),+) = self.parent.as_slices();
                unsafe {
                    Some(($($nm.get_unchecked(i)),+))
                }
            }

            fn size_hint(&self) -> (usize, Option<usize>) {
                (self.parent.len(), Some(self.parent.len()))
            }
        }

        pub struct $soazipm<'a, $($ty:'a),+> {
            $($nm: *mut $ty),+,
            tot: usize,
            i: usize
        }

        impl<'a, $($ty),+> Iterator for $soazipm<'a $($ty),+> {
            type Item = ($(&'a mut $ty),+);
            #[inline]
            fn next(&mut self) -> Option<($(&'a mut $ty),+)> {
                let i = self.i;
                if i == self.tot { return None }
                self.i += 1;
                unsafe {
                    let ($($nm),+) = ($(self.$nm),+);
                    // for zero-sized elements this will return the
                    // same pointer again, but that's ok since we only
                    // read from these pointers, never use them for
                    // size or to check if we're done.
                    $(self.$nm = self.$nm.offset(1));+; 
                    Some(($(mem::transmute($nm)),+))
                }
            }

            fn size_hint(&self) -> (usize, Option<usize>) {
                (self.tot, Some(self.tot))
            }
        }

        impl<$($ty),+> $soa<$($ty),+> {
            /// Constructs a new, empty `Soa2`, etc..
            ///
            /// The SoA will not allocate until elements are pushed onto it.
            pub fn new() -> $soa<$($ty),+> {
                unsafe {
                    $(let ($nm, $nmu) = Unadorned::new());+;
                    let e = unadorned::new_update(&[$($nmu),+]);
                    $soa { $($nm: $nm),+ , e: e}
                }
            }
            
            #[inline]
            /// Returns `true` if all our elements are zero-sized types.
            fn is_boring(&self) -> bool {
                $(self.$nm.is_boring())&&+
            }

            #[inline]
            /// Constructs a new, empty `SoA` with the specified capacity.
            ///
            /// The SoA will be able to hold exactly `capacity` tuples of elements
            /// without reallocating.
            ///
            /// If `capacity` is 0, the SoA will not allocate.
            ///
            /// It is important to note that this function does not
            /// specify the *length* of the soa, but only the
            /// *capacity*.
            pub fn with_capacity(capacity: usize) -> $soa<$($ty),+> {
                unsafe {
                    $(let ($nm, $nmu) = Unadorned::with_capacity(capacity));+;
                    let is_boring = $(mem::size_of::<$ty>() == 0)&&+;
                    let e = unadorned::with_capacity_update(&[$($nmu),+], is_boring, capacity);
                    $soa { $($nm: $nm),+, e: e}
                }
            }

            /// Constructs a `SoA` directly from the raw components of another.
            ///
            /// This is highly unsafe, and no invariants are checked.
            #[inline]
            pub unsafe fn from_raw_parts(
                $($nm: *mut $ty),+, len: usize, cap: usize) -> $soa<$($ty),+> {
                $(let ($nm, $nmu) = Unadorned::from_raw_parts($nm));+;
                let e = unadorned::from_raw_parts_update(&[$($nmu),+], len, cap);
                $soa { $($nm: $nm),+, e: e}
            }

            /// Constructs a `SoA` by copying the elements from raw pointers.
            ///
            /// This function will copy `elts` contiguous elements
            /// from each of the pointers into a new allocation owned
            /// by the returned `SoA`. The elements of the buffer are
            /// copied without cloning, as if `ptr::read()` were
            /// called on them.
            #[inline]
            pub unsafe fn from_raw_bufs($($nm: *const $ty),+, elts: usize) -> $soa<$($ty),+> {
                $(let ($nm, $nmu) = Unadorned::from_raw_bufs($nm, elts));+;
                let e = unadorned::from_raw_bufs_update(&[$($nmu),+], elts);
                $soa { $($nm: $nm),+, e: e}
            }

            /// Constructs a `SoA` directly from vectors of its components.
            ///
            /// This function will panic if the lengths of the vectors don't match.
            ///
            /// If the capacity of the vectors don't match they will be reallocated to
            /// have matching capacities.
            ///
            /// Otherwise, no allocation will be performed and the SoA will only take
            /// ownership of the elements in the vectors.
            pub fn from_vecs($(mut $nm: Vec<$ty>),+) -> $soa<$($ty),+> {
                let firstlen = first!($($nm.len()),+);
                if anyrest!($(firstlen != $nm.len()),+ ) {
                    panic!("unequal lengths");
                }
                let firstcap = first!($($nm.capacity()),+);
                if anyrest!($(firstcap != $nm.capacity()),+) {
                    $($nm.shrink_to_fit());+;
                }
                let cap = first!($($nm.capacity()),+);
                unsafe {
                    let ret = $soa::from_raw_parts(
                        $($nm.as_ptr() as *mut $ty),+,
                        firstlen, cap);
                    $(mem::forget($nm));+;
                    ret
                }
            }

            /// Returns the number of tuples stored in the SoA.
            #[inline]
            pub fn len(&self) -> usize { self.e.len }

            /// Returns `true` if the SoA contains no elements.
            #[inline]
            pub fn is_empty(&self) -> bool { self.len() == 0 }

            /// Sets the length of a vector.
            ///
            /// This will explicitly set the size of the soa, without actually
            /// modifying its buffers, so it is up to the caller to ensure that the
            /// SoA is actually the specified size.
            #[inline]
            pub unsafe fn set_len(&mut self, len: usize) {
                self.e.len = len;
            }

            /// Returns the number of elements the SoA can hold without reallocating.
            #[inline]
            pub fn capacity(&self) -> usize { self.e.cap }

            /// Reserves capacity for at least `additional` more
            /// elements to be inserted in the given SoA. The
            /// collection may reserve more space to avoid frequent
            /// reallocations.
            ///
            /// Panics if the new capacity overflows `usize`.
            pub fn reserve(&mut self, additional: usize) {
                let space = match unadorned::calc_reserve_space(&self.e, additional) {
                    None => return,
                    Some(space) => space
                };

                unsafe {
                    $(let $nmu = self.$nm.reserve(&self.e, &space));+;
                    unadorned::reserve_update(&[$($nmu),+], space, &mut self.e)
                }
            }

            /// Reserves the minimum capacity for exactly `additional` more elements to
            /// be inserted in the given SoA. Does nothing if the capacity is already
            /// sufficient.
            ///
            /// Note that the allocator may give the collection more space than it
            /// requests. Therefore, capacity can not be relied upon to be precisely
            /// minimal. Prefer `reserve` if future insertions are expected.
            ///
            /// Panics if the new capacity overflows `usize`.
            pub fn reserve_exact(&mut self, additional: usize) {
                let space = match unadorned::calc_reserve_exact_space(&self.e, additional) {
                    None => return,
                    Some(space) => space
                };

                unsafe {
                    $(let $nmu = self.$nm.reserve(&self.e, &space));+;
                    unadorned::reserve_update(&[$($nmu),+], space, &mut self.e)
                }
            }

            /// Shrinks the capacity of the SoA as much as possible.
            ///
            /// It will drop down as close as possible to the length, but the allocator
            /// may still inform the SoA that there is space for a few more elements.
            pub fn shrink_to_fit(&mut self) {
                if self.is_boring() { return }
                unsafe {
                    $(let $nmu = self.$nm.shrink_to_fit(&self.e));+;
                    unadorned::shrink_to_fit_update(&[$($nmu),+], &mut self.e);
                }
            }

            /// Shorten a SoA, dropping excess elements.
            ///
            /// If `len` is greater than the soa's current length, this has no effect.
            pub fn truncate(&mut self, len: usize) {
                if self.is_boring() { return }
                unsafe {
                    $(let $nmu = self.$nm.truncate(len, &self.e));+;
                    unadorned::truncate_update(&[$($nmu),+], len, &mut self.e);
                }
            }

            /// Returns mutable slices over the SoA's elements.            
            #[inline]
            pub fn as_mut_slices<'a> (&'a mut self) -> ($(&'a mut [$ty]),+) {
                unsafe {
                    let len = self.e.len;
                    ($(self.$nm.as_mut_slice(len)),+)
                }
            }

            /// Returns slices over the SoA's elements.
            #[inline]
            pub fn as_slices<'a>(&'a self) -> ($(&'a [$ty]),+) {
                unsafe { let len = self.e.len;
                         ($(self.$nm.as_slice(len)),+) }
            }

            /// Returns iterators over the SoA's elements.
            #[inline]
            pub fn iters(&self) -> ($(slice::Iter<$ty>),+) {
                let ($($nm),+) = self.as_slices();
                ($($nm.iter()),+)
            }

            /// Returns a single iterator over the SoA's elements, zipped up.
            #[inline]
            pub fn zip_iter<'a>(&'a self) -> $soazip<$($ty),+> {
                $soazip { parent: self, i: 0 }
            }
            
            /// Returns mutable iterators over the SoA's elements.
            #[inline]
            pub fn iters_mut(&mut self) -> ($(slice::IterMut<$ty>),+) {
                let ($($nm),+) = self.as_mut_slices();
                ($($nm.iter_mut()),+)
            }

            /// Returns a single iterator over the SoA's elements, zipped up.
            #[inline]
            pub fn zip_iter_mut(&mut self) -> $soazipm<$($ty),+> {
                unsafe {
                    $soazipm { 
                        $($nm: self.$nm.as_mut_slice(self.e.len).as_mut_ptr()),+, 
                        i: 0, 
                        tot: self.len()
                    }
                }
            }
            
            /// Converts an SoA into iterators for each of its arrays.
            #[inline]
            pub fn into_iters(mut self) -> ($(vec::IntoIter<$ty>),+) {
                unsafe {
                    let e_copy = self.e;
                    self.e.cap = 0;
                    ($(self.$nm.shallow_copy().into_iter(&e_copy)),+)
                }
            }

            /// Converts an SoA into a tuple of `Vec`s. This will neither allocator nor
            /// copy.
            #[inline]
            pub fn into_vecs(mut self) -> ($(Vec<$ty>),+) {
                unsafe {
                    let e_copy = self.e;
                    self.e.cap = 0;
                    ($(self.$nm.shallow_copy().as_vec(&e_copy)),+)
                }
            }

            /// Returns a tuple of pointers to the start of the data in an SoA.
            #[inline]
            pub fn as_ptrs(&self) -> ($(*const $ty),+) {
                let ($($nm),+) = self.as_slices();
                ($($nm.as_ptr()),+)
            }

            /// Returns a tuple of pointers to the start of the mutable data in an SoA.
            #[inline]
            pub fn as_mut_ptrs(&mut self) -> ($(*mut $ty),+) {
                let ($($nm),+) = self.as_mut_slices();
                ($($nm.as_mut_ptr()),+)
            }

            /// Removes an element from anywhere in the SoA and returns it, replacing it
            /// with the last element.
            ///
            /// This does not preserve ordering, but is O(1).
            ///
            /// Panics if `index` is out of bounds.
            #[inline]
            pub fn swap_remove(&mut self, index: usize) -> ($($ty),+) {
                let length = self.e.len;
                {
                    let ($($nm),+) = self.as_mut_slices();
                    $($nm.swap(index, length - 1));+;
                }
                self.pop().unwrap()
            }

            /// Inserts an element at position `index` within the vector, shifting all
            /// elements after position `index` one position to the right.
            ///
            /// Panics if `index` is not between `0` and the SoA's length, inclusive.
            pub fn insert(&mut self, index: usize, element: ($($ty),+)) {
                unsafe {
                    assert!(index < self.e.len);
                    let space = unadorned::calc_reserve_space(&self.e, 1);
                    let ($($nm),+) = element;
                    $(let $nmu = self.$nm.insert(index, $nm, &self.e, &space));+;
                    unadorned::insert_update(&[$($nmu),+], space, &mut self.e);
                }
            }

            /// Removes and returns the elements at position `index` within the SoA,
            /// shifting all elements after position `index` one position to the left.
            ///
            /// Panics if `index` is out of bounds.
            pub fn remove(&mut self, index: usize) -> ($($ty),+) {
                unsafe {
                    assert!(index < self.e.len);
                    $(let ($nm, $nmu) = self.$nm.remove(index, &self.e));+;
                    unadorned::remove_update(&[$($nmu),+], &mut self.e);
                    ($($nm),+)
                }
            }
            
            /// Returns only the element specified by the predicate.
            ///
            /// In other words, remove all elements `e` such that `f(&e)` returns false.
            /// This method operates in place and preserves the order of the retained
            /// elements.
            pub fn retain<F_>(&mut self, mut f: F_) 
                where F_: FnMut(($(&$ty),+)) -> bool {
                let len = self.len();
                let mut del = 0us;

                {
                    let ($($nm),+) = self.as_mut_slices();
                    for i in range(0us, len) {
                        if !f(($(&$nm[i]),+)) {
                            del += 1;
                        } else if del > 0 {
                            $($nm.swap(i-del, i));+;
                        }
                    }
                }

                self.truncate(len - del);
            }

            /// Appends an element to the back of a collection.
            ///
            /// Panics if the number of elements in the SoA overflows a `usize`.
            #[inline]
            pub fn push(&mut self, value: ($($ty),+)) {
                if self.is_boring() {
                    self.e.len = self.e.len.checked_add(1).expect("length overflow");
                    unsafe { mem::forget(value) }
                    return
                }
                unsafe {
                    let ($($nm),+) = value;
                    $(let $nmu = self.$nm.push($nm, &self.e));+;
                    unadorned::push_update(&[$($nmu),+], &mut self.e);
                }
            }

            /// Removes the last element from a SoA and returns it, or `None` if empty.
            #[inline]
            pub fn pop(&mut self) -> Option<($($ty),+)> {
                if self.e.len == 0 { 
                    None 
                } else {
                    unsafe {
                        self.e.len -= 1;
                        let len = self.e.len;
                        let ($($nm),+) = self.as_mut_slices();
                        Some(($(ptr::read($nm.get_unchecked(len))),+))
                    }
                }
            }

            /// Moves all the elements of `other` into `self`, leaving `other` empty.
            ///
            /// Panics if the number of elements in the SoA overflows a `usize`.
            #[inline]
            pub fn append(&mut self, other: &mut Self) {
                if self.is_boring() {
                    self.e.len = self.e.len.checked_add(other.len()).expect("length overflow");
                    other.e.len = 0;
                    return ;
                }
                unsafe {
                    let space = unadorned::calc_reserve_space(&self.e, 1);
                    $(let $nmu = self.$nm.append(&self.e, &other.$nm, &other.e, &space));+;
                    unadorned::append_update(&[$($nmu),+], &mut self.e, &mut other.e, space);
                }
            }

            // TODO: drain

            /// Clears the SoA, removing all values.
            #[inline]
            pub fn clear(&mut self) { self.truncate(0) }

            // TODO: map_in_place

            /// Extends the SoA with the elements yielded by arbitrary iterators.
            ///
            /// Panics (and leaks memory!) if the iterators yield a different number of
            /// elements.
            #[allow(non_camel_case_types)]
            pub fn extend<$($nm),+>(&mut self, $($nm: $nm),+) 
                where $($nm: Iterator<Item=$ty>),+
            {
                unsafe {
                    let (lower, _) = first!($($nm.size_hint()),+);
                    let space = unadorned::calc_reserve_space(&self.e, lower);
                    $(let $nmu = self.$nm.extend(&self.e, &space, $nm));+;
                    unadorned::extend_update(&[$($nmu),+], &mut self.e);
                }
            }

            /// Constructs an `SoA` with elements yielded by arbitrary iterators.
            ///
            /// Panics (and leaks memory!) if the iterators yield a different number of
            /// elements.
            #[allow(non_camel_case_types)]
            pub fn from_iters<$($nm),+>($($nm:$nm),+) -> $soa<$($ty),+> 
                where $($nm: Iterator<Item=$ty>),+ 
            {
                let mut v = $soa::new();
                v.extend($($nm),+);
                v
            }

            // TODO: dedup
        }

        impl<$($ty: Clone),+> $soa<$($ty),+> {
            /// Resizes the SoA in-place so that `len()` is equal to `new_len`.
            ///
            /// Calls either `extend()` or `truncate()` depending on whether `new_len` is
            /// larger than the current value of `len()` or not.
            #[inline]
            pub fn resize(&mut self, new_len: usize, value: ($($ty),+)) {
                let len = self.len();
                if new_len > len {
                    let ($($nm),+) = value;
                    self.extend($(repeat($nm).take(new_len - len)),+);
                } else {
                    self.truncate(new_len)
                }
            }

            /// Appends all elements in slices to the SoA.
            ///
            /// Iterates over the slices, clones each element, and then appends them to
            /// this SoA. The slices are traversed one at a time, in order.
            ///
            /// Panics if the slices are of different lengths.
            #[inline]
            pub fn push_all(&mut self, $($nm: &[$ty]),+) {
                unsafe {
                    let firstlen = first!($($nm.len()),+);
                    if anyrest!($(firstlen != $nm.len()),+) {
                        panic!("lengths not equal")
                    }
                    let space = unadorned::calc_reserve_space(&self.e, firstlen);
                    $(let $nmu = self.$nm.push_all($nm, &self.e, &space));+;
                    unadorned::push_all_update(&[$($nmu),+], &mut self.e, firstlen, space);
                }
            }
        }

        impl<$($ty: Clone),+> Clone for $soa<$($ty),+> {
            #[inline]
            fn clone(&self) -> $soa<$($ty),+> {
                let mut ret = $soa::new();
                let ($($nm),+) = self.as_slices();
                ret.push_all($($nm),+);
                ret
            }

            fn clone_from(&mut self, other: &$soa<$($ty),+>) {
                // TODO: cleanup
                
                if self.len() > other.len() {
                    self.truncate(other.len());
                }
                
                let ($($nmu),+) = other.as_slices();
                
                let ($($nmu),+) = {
                    let self_len = self.len();
                    let ($($nm),+) = self.iters_mut();
                    
                    $(for (place, thing) in $nm.zip($nmu.iter()) {
                        place.clone_from(thing);
                    })+;
                    
                    $(let $nm = &$nmu[self_len..]);+;
                    
                    ($($nm),+)
                };
                
                self.push_all($($nmu),+)
            }
        }

        impl<S: hash::Writer + hash::Hasher, $($ty: Hash<S>),+> Hash<S> for $soa<$($ty),+> {
            #[inline]
            fn hash(&self, state: &mut S) {
                self.as_slices().hash(state)
            }
        }

        #[allow(non_camel_case_types)]
        impl<$($nm),+, $($nmu),+> PartialEq<$soa<$($nmu),+>> for $soa<$($nm),+>
            where $($nm: PartialEq<$nmu>),+ 
        {
            #[inline]
            fn eq(&self, other: &$soa<$($nmu),+>) -> bool {
                let ($($nm),+) = self.as_slices();
                let ($($nmu),+) = other.as_slices();
                $(PartialEq::eq($nm, $nmu))&&+
            }
            #[inline]
            fn ne(&self, other: &$soa<$($nmu),+>) -> bool {
                let ($($nm),+) = self.as_slices();
                let ($($nmu),+) = other.as_slices();
                $(PartialEq::ne($nm, $nmu))||+
            }
        }

        #[allow(non_camel_case_types)]
        impl<$($nm),+, $($nmu),+> PartialEq<Vec<($($nmu),+)>> for $soa<$($nm),+>
            where $($nm: PartialEq<$nmu>),+ 
        {
            #[inline]
            fn eq(&self, other: &Vec<($($nmu),+)>) -> bool {
                self.len() == other.len() 
                && self.zip_iter().zip(other.iter()).all(
                    |(($($nm),+), &($(ref $nmu),+))|
                    $($nm == $nmu)&&+)
            }

            #[inline]
            fn ne(&self, other: &Vec<($($nmu),+)>) -> bool {
                self.len() != other.len() 
                    || self.zip_iter().zip(other.iter()).any(
                        |(($($nm),+), &($(ref $nmu),+))| $($nm != $nmu)||+)
            }
        }

        #[allow(non_camel_case_types)]
        impl<'b, $($nm),+, $($nmu),+> PartialEq<&'b [($($nmu),+)]> for $soa<$($nm),+>
            where $($nm: PartialEq<$nmu>),+ 
        {
            #[inline]
            fn eq(&self, other: &&'b [($($nmu),+)]) -> bool {
                self.len() == other.len() 
                    && self.zip_iter().zip(other.iter()).all(
                        |(($($nm),+), &($(ref $nmu),+))|
                        $($nm == $nmu)&&+)
            }
            
            #[inline]
            fn ne(&self, other: &&'b [($($nmu),+)]) -> bool {
                self.len() != other.len() 
                    || self.zip_iter().zip(other.iter()).any(
                        |(($($nm),+), &($(ref $nmu),+))| $($nm != $nmu)||+)
            }
        }

        #[allow(non_camel_case_types)]
        impl<'b, $($nm),+, $($nmu),+> PartialEq<&'b mut [($($nmu),+)]> for $soa<$($nm),+>
            where $($nm: PartialEq<$nmu>),+ 
        {
            #[inline]
            fn eq(&self, other: &&'b mut [($($nmu),+)]) -> bool {
                self.len() == other.len() 
                    && self.zip_iter().zip(other.iter()).all(
                        |(($($nm),+), &($(ref $nmu),+))|
                        $($nm == $nmu)&&+)
            }
            
            #[inline]
            fn ne(&self, other: &&'b mut [($($nmu),+)]) -> bool {
                self.len() != other.len() 
                    || self.zip_iter().zip(other.iter()).any(
                        |(($($nm),+), &($(ref $nmu),+))| $($nm != $nmu)||+)
            }
        }

        impl<$($ty: Eq),+> Eq for $soa<$($ty),+> {}

        impl<$($ty: PartialOrd),+> PartialOrd for $soa<$($ty),+> {
            #[inline]
            fn partial_cmp(&self, other: &$soa<$($ty),+>) -> Option<Ordering> {
                iter::order::partial_cmp(self.zip_iter(), other.zip_iter())
            }
        }

        impl<$($ty: Ord),+> Ord for $soa<$($ty),+> {
            #[inline]
            fn cmp(&self, other: &$soa<$($ty),+>) -> Ordering {
                iter::order::cmp(self.zip_iter(), other.zip_iter())
            }
        }

        impl<$($ty),+> Default for $soa<$($ty),+> {
            fn default() -> $soa<$($ty),+> { $soa::new() }
        }

        impl<$($ty: Debug),+> Debug for $soa<$($ty),+> {
            fn fmt(&self, f: &mut Formatter) -> Result {
                Debug::fmt(&self.as_slices(), f)
            }
        }

        #[unsafe_destructor]
        impl<$($ty),+> Drop for $soa<$($ty),+> {
            #[inline]
            fn drop(&mut self) {
                if self.e.cap != 0 {
                    unsafe {
                        $(self.$nm.drop(&self.e));+;
                    }
                }
            }
        }
    }
}

gen_soa!(Soa2, Soa2Zip, Soa2ZipM | T0,T1 | d0,d1 | d0u,d1u);
gen_soa!(Soa3, Soa3Zip, Soa3ZipM | T0,T1,T2 | d0,d1,d2 | d0u,d1u,d2u);
gen_soa!(Soa4, Soa4Zip, Soa4ZipM | T0,T1,T2,T3| d0,d1,d2,d3 | d0u,d1u,d2u,d3u);
gen_soa!(Soa5, Soa5Zip, Soa5ZipM | T0,T1,T2,T3,T4| d0,d1,d2,d3,d4 | d0u,d1u,d2u,d3u,d4u);
gen_soa!(Soa6, Soa6Zip, Soa6ZipM | T0,T1,T2,T3,T4,T5| d0,d1,d2,d3,d4,d5 | d0u,d1u,d2u,d3u,d4u,d5u);
gen_soa!(Soa7, Soa7Zip, Soa7ZipM | T0,T1,T2,T3,T4,T5,T6| d0,d1,d2,d3,d4,d5,d6 | d0u,d1u,d2u,d3u,d4u,d5u,d6u);
gen_soa!(Soa8, Soa8Zip, Soa8ZipM | T0,T1,T2,T3,T4,T5,T6,T7| d0,d1,d2,d3,d4,d5,d6,d7 | d0u,d1u,d2u,d3u,d4u,d5u,d6u,d7u);
gen_soa!(Soa9, Soa9Zip, Soa9ZipM | T0,T1,T2,T3,T4,T5,T6,T7,T8| d0,d1,d2,d3,d4,d5,d6,d7,d8 | d0u,d1u,d2u,d3u,d4u,d5u,d6u,d7u,d8u);
gen_soa!(Soa10, Soa10Zip, Soa10ZipM | T0,T1,T2,T3,T4,T5,T6,T7,T8,T9| d0,d1,d2,d3,d4,d5,d6,d7,d8,d9 | d0u,d1u,d2u,d3u,d4u,d5u,d6u,d7u,d8u,d9u);
gen_soa!(Soa11, Soa11Zip, Soa11ZipM | T0,T1,T2,T3,T4,T5,T6,T7,T8,T9,T10| d0,d1,d2,d3,d4,d5,d6,d7,d8,d9,d10 | d0u,d1u,d2u,d3u,d4u,d5u,d6u,d7u,d8u,d9u,d10u);
// we can't kick the can down to other trait implementations---e.g. of
// Hash---for these large tuples. Oh well!
// gen_soa!(Soa12, Soa12Zip, Soa12ZipM | T0,T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11| d0,d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11 | d0u,d1u,d2u,d3u,d4u,d5u,d6u,d7u,d8u,d9u,d10u,d11u);
// gen_soa!(Soa13, Soa13Zip, Soa13ZipM | T0,T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12| d0,d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12 | d0u,d1u,d2u,d3u,d4u,d5u,d6u,d7u,d8u,d9u,d10u,d11u,d12u);
// gen_soa!(Soa14, Soa14Zip, Soa14ZipM | T0,T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13| d0,d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13 | d0u,d1u,d2u,d3u,d4u,d5u,d6u,d7u,d8u,d9u,d10u,d11u,d12u,d13u);
// gen_soa!(Soa15, Soa15Zip, Soa15ZipM | T0,T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14| d0,d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13,d14 | d0u,d1u,d2u,d3u,d4u,d5u,d6u,d7u,d8u,d9u,d10u,d11u,d12u,d13u,d14u);
// gen_soa!(Soa16, Soa16Zip, Soa16ZipM | T0,T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15| d0,d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13,d14,d15 | d0u,d1u,d2u,d3u,d4u,d5u,d6u,d7u,d8u,d9u,d10u,d11u,d12u,d13u,d14u,d15u);
// gen_soa!(Soa17, Soa17Zip, Soa17ZipM | T0,T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16| d0,d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13,d14,d15,d16 | d0u,d1u,d2u,d3u,d4u,d5u,d6u,d7u,d8u,d9u,d10u,d11u,d12u,d13u,d14u,d15u,d16u);
// gen_soa!(Soa18, Soa18Zip, Soa18ZipM | T0,T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16,T17| d0,d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13,d14,d15,d16,d17 | d0u,d1u,d2u,d3u,d4u,d5u,d6u,d7u,d8u,d9u,d10u,d11u,d12u,d13u,d14u,d15u,d16u,d17u);
// gen_soa!(Soa19, Soa19Zip, Soa19ZipM | T0,T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16,T17,T18| d0,d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13,d14,d15,d16,d17,d18 | d0u,d1u,d2u,d3u,d4u,d5u,d6u,d7u,d8u,d9u,d10u,d11u,d12u,d13u,d14u,d15u,d16u,d17u,d18u);
// gen_soa!(Soa20, Soa20Zip, Soa20ZipM | T0,T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16,T17,T18,T19| d0,d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13,d14,d15,d16,d17,d18,d19 | d0u,d1u,d2u,d3u,d4u,d5u,d6u,d7u,d8u,d9u,d10u,d11u,d12u,d13u,d14u,d15u,d16u,d17u,d18u,d19u);
// gen_soa!(Soa21, Soa21Zip, Soa21ZipM | T0,T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16,T17,T18,T19,T20| d0,d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13,d14,d15,d16,d17,d18,d19,d20 | d0u,d1u,d2u,d3u,d4u,d5u,d6u,d7u,d8u,d9u,d10u,d11u,d12u,d13u,d14u,d15u,d16u,d17u,d18u,d19u,d20u);
// gen_soa!(Soa22, Soa22Zip, Soa22ZipM | T0,T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16,T17,T18,T19,T20,T21| d0,d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13,d14,d15,d16,d17,d18,d19,d20,d21 | d0u,d1u,d2u,d3u,d4u,d5u,d6u,d7u,d8u,d9u,d10u,d11u,d12u,d13u,d14u,d15u,d16u,d17u,d18u,d19u,d20u,d21u);
// gen_soa!(Soa23, Soa23Zip, Soa23ZipM | T0,T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16,T17,T18,T19,T20,T21,T22| d0,d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13,d14,d15,d16,d17,d18,d19,d20,d21,d22 | d0u,d1u,d2u,d3u,d4u,d5u,d6u,d7u,d8u,d9u,d10u,d11u,d12u,d13u,d14u,d15u,d16u,d17u,d18u,d19u,d20u,d21u,d22u);
// gen_soa!(Soa24, Soa24Zip, Soa24ZipM | T0,T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16,T17,T18,T19,T20,T21,T22,T23| d0,d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13,d14,d15,d16,d17,d18,d19,d20,d21,d22,d23 | d0u,d1u,d2u,d3u,d4u,d5u,d6u,d7u,d8u,d9u,d10u,d11u,d12u,d13u,d14u,d15u,d16u,d17u,d18u,d19u,d20u,d21u,d22u,d23u);
// gen_soa!(Soa25, Soa25Zip, Soa25ZipM | T0,T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16,T17,T18,T19,T20,T21,T22,T23,T24| d0,d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13,d14,d15,d16,d17,d18,d19,d20,d21,d22,d23,d24 | d0u,d1u,d2u,d3u,d4u,d5u,d6u,d7u,d8u,d9u,d10u,d11u,d12u,d13u,d14u,d15u,d16u,d17u,d18u,d19u,d20u,d21u,d22u,d23u,d24u);
// gen_soa!(Soa26, Soa26Zip, Soa26ZipM | T0,T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16,T17,T18,T19,T20,T21,T22,T23,T24,T25| d0,d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13,d14,d15,d16,d17,d18,d19,d20,d21,d22,d23,d24,d25 | d0u,d1u,d2u,d3u,d4u,d5u,d6u,d7u,d8u,d9u,d10u,d11u,d12u,d13u,d14u,d15u,d16u,d17u,d18u,d19u,d20u,d21u,d22u,d23u,d24u,d25u);
// gen_soa!(Soa27, Soa27Zip, Soa27ZipM | T0,T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16,T17,T18,T19,T20,T21,T22,T23,T24,T25,T26| d0,d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13,d14,d15,d16,d17,d18,d19,d20,d21,d22,d23,d24,d25,d26 | d0u,d1u,d2u,d3u,d4u,d5u,d6u,d7u,d8u,d9u,d10u,d11u,d12u,d13u,d14u,d15u,d16u,d17u,d18u,d19u,d20u,d21u,d22u,d23u,d24u,d25u,d26u);
// gen_soa!(Soa28, Soa28Zip, Soa28ZipM | T0,T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16,T17,T18,T19,T20,T21,T22,T23,T24,T25,T26,T27| d0,d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13,d14,d15,d16,d17,d18,d19,d20,d21,d22,d23,d24,d25,d26,d27 | d0u,d1u,d2u,d3u,d4u,d5u,d6u,d7u,d8u,d9u,d10u,d11u,d12u,d13u,d14u,d15u,d16u,d17u,d18u,d19u,d20u,d21u,d22u,d23u,d24u,d25u,d26u,d27u);
// gen_soa!(Soa29, Soa29Zip, Soa29ZipM | T0,T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16,T17,T18,T19,T20,T21,T22,T23,T24,T25,T26,T27,T28| d0,d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13,d14,d15,d16,d17,d18,d19,d20,d21,d22,d23,d24,d25,d26,d27,d28 | d0u,d1u,d2u,d3u,d4u,d5u,d6u,d7u,d8u,d9u,d10u,d11u,d12u,d13u,d14u,d15u,d16u,d17u,d18u,d19u,d20u,d21u,d22u,d23u,d24u,d25u,d26u,d27u,d28u);
// gen_soa!(Soa30, Soa30Zip, Soa30ZipM | T0,T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16,T17,T18,T19,T20,T21,T22,T23,T24,T25,T26,T27,T28,T29| d0,d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13,d14,d15,d16,d17,d18,d19,d20,d21,d22,d23,d24,d25,d26,d27,d28,d29 | d0u,d1u,d2u,d3u,d4u,d5u,d6u,d7u,d8u,d9u,d10u,d11u,d12u,d13u,d14u,d15u,d16u,d17u,d18u,d19u,d20u,d21u,d22u,d23u,d24u,d25u,d26u,d27u,d28u,d29u);
// gen_soa!(Soa31, Soa31Zip, Soa31ZipM | T0,T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16,T17,T18,T19,T20,T21,T22,T23,T24,T25,T26,T27,T28,T29,T30| d0,d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13,d14,d15,d16,d17,d18,d19,d20,d21,d22,d23,d24,d25,d26,d27,d28,d29,d30 | d0u,d1u,d2u,d3u,d4u,d5u,d6u,d7u,d8u,d9u,d10u,d11u,d12u,d13u,d14u,d15u,d16u,d17u,d18u,d19u,d20u,d21u,d22u,d23u,d24u,d25u,d26u,d27u,d28u,d29u,d30u);
