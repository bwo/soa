//! Growable struct-of-array types with heap allocated contents.
#![allow(unused_features)]

#![feature(alloc)]
#![feature(collections)]
#![feature(core)]
#![feature(hash)]
#![feature(test)]

#![feature(unsafe_destructor)]

extern crate alloc;
extern crate collections;
extern crate core;

pub mod soa;

mod unadorned;
#[cfg(test)] mod test;

pub use soa::{Soa2,Soa3,Soa4,Soa5,Soa6,Soa7,Soa8,Soa9,Soa10,Soa11};
