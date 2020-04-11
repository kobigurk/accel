use super::*;

/// Determine actual memory type dynamically
///
/// Because `Continuous` memories can be treated as a slice,
/// input slice may represents any type of memory.
fn memory_type<T>(ptr: *const T) -> MemoryType {
    match get_attr(ptr, CUpointer_attribute::CU_POINTER_ATTRIBUTE_MEMORY_TYPE) {
        Ok(CUmemorytype_enum::CU_MEMORYTYPE_HOST) => MemoryType::PageLocked,
        Ok(CUmemorytype_enum::CU_MEMORYTYPE_DEVICE) => MemoryType::Device,
        Ok(CUmemorytype_enum::CU_MEMORYTYPE_ARRAY) => MemoryType::Array,
        Ok(CUmemorytype_enum::CU_MEMORYTYPE_UNIFIED) => MemoryType::Registered,
        Err(_) => MemoryType::Host,
    }
}

impl<'a, T: Copy> Memory for &'a [T] {
    type Elem = T;
    fn head_addr(&self) -> *const T {
        self.as_ptr()
    }
    fn byte_size(&self) -> usize {
        self.len() * std::mem::size_of::<T>()
    }
    fn try_as_slice(&self) -> Result<&[T]> {
        Ok(self)
    }
    fn try_get_context(&self) -> Option<&Context> {
        None
    }
    fn memory_type(&self) -> MemoryType {
        memory_type(self.as_ptr())
    }
}

impl<'a, T: Copy> Memory for &'a mut [T] {
    type Elem = T;
    fn head_addr(&self) -> *const T {
        self.as_ptr()
    }
    fn byte_size(&self) -> usize {
        self.len() * std::mem::size_of::<T>()
    }
    fn try_as_slice(&self) -> Result<&[T]> {
        Ok(self)
    }
    fn try_get_context(&self) -> Option<&Context> {
        None
    }
    fn memory_type(&self) -> MemoryType {
        memory_type(self.as_ptr())
    }
}

impl<'a, T: Copy> MemoryMut for &'a mut [T] {
    fn head_addr_mut(&mut self) -> *mut T {
        self.as_mut_ptr()
    }
    fn try_as_mut_slice(&mut self) -> Result<&mut [T]> {
        Ok(self)
    }
    fn copy_from(&mut self, src: &impl Memory<Elem = Self::Elem>) {
        assert_ne!(self.head_addr(), src.head_addr());
        assert_eq!(self.byte_size(), src.byte_size());

        // Tuple (dest, src) cannot be matched by or-patterns syntax, which is experimental
        // See https://github.com/rust-lang/rust/issues/54883
        match self.memory_type() {
            // To host
            MemoryType::Host | MemoryType::Registered | MemoryType::PageLocked => unsafe {
                copy_to_host(self, src)
            },
            // To device
            MemoryType::Device => unsafe { copy_to_device(self, src) },
            // To array
            MemoryType::Array => unimplemented!("Array memory is not supported yet"),
        }
    }
}

impl<'a, T: Copy> Continuous for &'a [T] {
    fn length(&self) -> usize {
        self.len()
    }
    fn as_slice(&self) -> &[Self::Elem] {
        self
    }
}

impl<'a, T: Copy> Continuous for &'a mut [T] {
    fn length(&self) -> usize {
        self.len()
    }
    fn as_slice(&self) -> &[Self::Elem] {
        self
    }
}

impl<'a, T: Copy> ContinuousMut for &'a mut [T] {
    fn as_mut_slice(&mut self) -> &mut [Self::Elem] {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_for_slice() -> Result<()> {
        let a = vec![0_u32; 12];
        assert!(matches!(a.as_slice().memory_type(), MemoryType::Host));
        assert_eq!(a.as_slice().length(), 12);
        assert_eq!(a.as_slice().byte_size(), 12 * 4 /* u32 */);
        Ok(())
    }
}
