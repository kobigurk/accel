//! GPGPU framework for Rust

extern crate cuda_driver_sys as cuda;

pub mod array;
pub mod device;
pub mod error;
pub mod instruction;
pub mod linker;
pub mod memory;
pub mod module;
pub mod stream;

pub use device::{Context, Device};
use num_traits::ToPrimitive;

/// Size of Block (thread block) in [CUDA thread hierarchy]( http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-model )
///
/// Every input integer and float convert into `u32` using [ToPrimitive].
/// If the conversion is impossible, e.g. negative or too large integers, the conversion will panics.
///
/// [ToPrimitive]: https://docs.rs/num-traits/0.2.11/num_traits/cast/trait.ToPrimitive.html
///
/// Examples
/// --------
///
/// - Explicit creation
///
/// ```
/// # use accel::*;
/// let block1d = Block::x(64);
/// assert_eq!(block1d.x, 64);
///
/// let block2d = Block::xy(64, 128);
/// assert_eq!(block2d.x, 64);
/// assert_eq!(block2d.y, 128);
///
/// let block3d = Block::xyz(64, 128, 256);
/// assert_eq!(block3d.x, 64);
/// assert_eq!(block3d.y, 128);
/// assert_eq!(block3d.z, 256);
/// ```
///
/// - From single integer (unsigned and signed)
///
/// ```
/// # use accel::*;
/// let block1d: Block = 64_usize.into();
/// assert_eq!(block1d.x, 64);
///
/// let block1d: Block = 64_i32.into();
/// assert_eq!(block1d.x, 64);
/// ```
///
/// - From tuple
///
/// ```
/// # use accel::*;
/// let block1d: Block = (64,).into();
/// assert_eq!(block1d.x, 64);
///
/// let block2d: Block = (64, 128).into();
/// assert_eq!(block2d.x, 64);
/// assert_eq!(block2d.y, 128);
///
/// let block3d: Block = (64, 128, 256).into();
/// assert_eq!(block3d.x, 64);
/// assert_eq!(block3d.y, 128);
/// assert_eq!(block3d.z, 256);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Block {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl Block {
    /// 1D Block
    ///
    /// Panic
    /// -----
    /// - If input values cannot convert to u32
    pub fn x<I: ToPrimitive>(x: I) -> Self {
        Block {
            x: x.to_u32().expect("Cannot convert to u32"),
            y: 1,
            z: 1,
        }
    }

    /// 2D Block
    ///
    /// Panic
    /// -----
    /// - If input values cannot convert to u32
    pub fn xy<I1: ToPrimitive, I2: ToPrimitive>(x: I1, y: I2) -> Self {
        Block {
            x: x.to_u32().expect("Cannot convert to u32"),
            y: y.to_u32().expect("Cannot convert to u32"),
            z: 1,
        }
    }

    /// 3D Block
    ///
    /// Panic
    /// -----
    /// - If input values cannot convert to u32
    pub fn xyz<I1: ToPrimitive, I2: ToPrimitive, I3: ToPrimitive>(x: I1, y: I2, z: I3) -> Self {
        Block {
            x: x.to_u32().expect("Cannot convert to u32"),
            y: y.to_u32().expect("Cannot convert to u32"),
            z: z.to_u32().expect("Cannot convert to u32"),
        }
    }
}

impl<I: ToPrimitive> Into<Block> for (I,) {
    fn into(self) -> Block {
        Block::x(self.0)
    }
}

impl<I1: ToPrimitive, I2: ToPrimitive> Into<Block> for (I1, I2) {
    fn into(self) -> Block {
        Block::xy(self.0, self.1)
    }
}

impl<I1: ToPrimitive, I2: ToPrimitive, I3: ToPrimitive> Into<Block> for (I1, I2, I3) {
    fn into(self) -> Block {
        Block::xyz(self.0, self.1, self.2)
    }
}

macro_rules! impl_into_block {
    ($integer:ty) => {
        impl Into<Block> for $integer {
            fn into(self) -> Block {
                Block::x(self)
            }
        }
    };
}

impl_into_block!(u8);
impl_into_block!(u16);
impl_into_block!(u32);
impl_into_block!(u64);
impl_into_block!(u128);
impl_into_block!(usize);
impl_into_block!(i8);
impl_into_block!(i16);
impl_into_block!(i32);
impl_into_block!(i64);
impl_into_block!(i128);
impl_into_block!(isize);

/// Size of Grid (grid of blocks) in [CUDA thread hierarchy]( http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-model )
///
/// Every input integer and float convert into `u32` using [ToPrimitive].
/// If the conversion is impossible, e.g. negative or too large integers, the conversion will panics.
///
/// [ToPrimitive]: https://docs.rs/num-traits/0.2.11/num_traits/cast/trait.ToPrimitive.html
///
/// Examples
/// --------
///
/// - Explicit creation
///
/// ```
/// # use accel::*;
/// let grid1d = Grid::x(64);
/// assert_eq!(grid1d.x, 64);
///
/// let grid2d = Grid::xy(64, 128);
/// assert_eq!(grid2d.x, 64);
/// assert_eq!(grid2d.y, 128);
///
/// let grid3d = Grid::xyz(64, 128, 256);
/// assert_eq!(grid3d.x, 64);
/// assert_eq!(grid3d.y, 128);
/// assert_eq!(grid3d.z, 256);
/// ```
///
/// - From single integer (unsigned and signed)
///
/// ```
/// # use accel::*;
/// let grid1d: Grid = 64_usize.into();
/// assert_eq!(grid1d.x, 64);
///
/// let grid1d: Grid = 64_i32.into();
/// assert_eq!(grid1d.x, 64);
/// ```
///
/// - From tuple
///
/// ```
/// # use accel::*;
/// let grid1d: Grid = (64,).into();
/// assert_eq!(grid1d.x, 64);
///
/// let grid2d: Grid = (64, 128).into();
/// assert_eq!(grid2d.x, 64);
/// assert_eq!(grid2d.y, 128);
///
/// let grid3d: Grid = (64, 128, 256).into();
/// assert_eq!(grid3d.x, 64);
/// assert_eq!(grid3d.y, 128);
/// assert_eq!(grid3d.z, 256);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Grid {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl Grid {
    /// 1D Grid
    ///
    /// Panic
    /// -----
    /// - If input values cannot convert to u32
    pub fn x<I: ToPrimitive>(x: I) -> Self {
        Grid {
            x: x.to_u32().expect("Cannot convert to u32"),
            y: 1,
            z: 1,
        }
    }

    /// 2D Grid
    ///
    /// Panic
    /// -----
    /// - If input values cannot convert to u32
    pub fn xy<I1: ToPrimitive, I2: ToPrimitive>(x: I1, y: I2) -> Self {
        Grid {
            x: x.to_u32().expect("Cannot convert to u32"),
            y: y.to_u32().expect("Cannot convert to u32"),
            z: 1,
        }
    }

    /// 3D Grid
    ///
    /// Panic
    /// -----
    /// - If input values cannot convert to u32
    pub fn xyz<I1: ToPrimitive, I2: ToPrimitive, I3: ToPrimitive>(x: I1, y: I2, z: I3) -> Self {
        Grid {
            x: x.to_u32().expect("Cannot convert to u32"),
            y: y.to_u32().expect("Cannot convert to u32"),
            z: z.to_u32().expect("Cannot convert to u32"),
        }
    }
}

impl<I: ToPrimitive> Into<Grid> for (I,) {
    fn into(self) -> Grid {
        Grid::x(self.0)
    }
}

impl<I1: ToPrimitive, I2: ToPrimitive> Into<Grid> for (I1, I2) {
    fn into(self) -> Grid {
        Grid::xy(self.0, self.1)
    }
}

impl<I1: ToPrimitive, I2: ToPrimitive, I3: ToPrimitive> Into<Grid> for (I1, I2, I3) {
    fn into(self) -> Grid {
        Grid::xyz(self.0, self.1, self.2)
    }
}

macro_rules! impl_into_grid {
    ($integer:ty) => {
        impl Into<Grid> for $integer {
            fn into(self) -> Grid {
                Grid::x(self)
            }
        }
    };
}

impl_into_grid!(u8);
impl_into_grid!(u16);
impl_into_grid!(u32);
impl_into_grid!(u64);
impl_into_grid!(u128);
impl_into_grid!(usize);
impl_into_grid!(i8);
impl_into_grid!(i16);
impl_into_grid!(i32);
impl_into_grid!(i64);
impl_into_grid!(i128);
impl_into_grid!(isize);
