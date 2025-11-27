/*
use crate::protocols::errors::SvsmReqError;
use crate::mm::PerCPUPageMappingGuard;
use crate::types::PAGE_SIZE;
use crate::cpu::percpu::this_cpu;
use crate::address::{PhysAddr, Address};
*/

// const SHMEM_MAPPING_SIZE: u64 = 0x2000;
#[derive(Debug)]
#[repr(packed)]
pub struct SharedMemoryData {
    pub address: u64,
    _size: u64,
    _buf: [u64; 510],
    _value: [u64; 512],
}

/*
struct SharedMemory {
    shmem: *mut SharedMemoryData,
    guard: PerCPUPageMappingGuard,
    phys_addr: PhysAddr,
    size: u64,
}

impl SharedMemory {
    fn new(size: u64, phys_addr: u64) -> Result<SharedMemory, SvsmReqError> {
        if size < SHMEM_MAPPING_SIZE {
            log::info!("shared memory size is not enough.");
            return Err(SvsmReqError::invalid_parameter());
        }
        let phys_addr = PhysAddr::from(phys_addr & 0x0000_ffff_ffff_ffff);
        let offset = phys_addr.page_offset();
        let guard = PerCPUPageMappingGuard::create(
                phys_addr, phys_addr + PAGE_SIZE * 2, 0)?;
        let virt_addr = guard.virt_addr();
        this_cpu()
            .get_pgtable()
            .set_shared_4k(virt_addr + offset)?;
        this_cpu()
            .get_pgtable()
            .set_shared_4k(virt_addr + offset + PAGE_SIZE)?;

        Ok(SharedMemory {
            shmem: virt_addr.as_mut_ptr::<SharedMemoryData>(),
            guard: guard,
            phys_addr: phys_addr,
            size: size,
        })
    }
}
*/
