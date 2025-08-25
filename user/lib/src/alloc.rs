use core::ptr;
use core::cell::UnsafeCell;
use core::alloc::GlobalAlloc;
extern crate alloc;
use alloc::alloc::Layout;

struct SimpleAlloc {
	head: UnsafeCell<usize>,
	end: usize,
}

unsafe impl Sync for SimpleAlloc {}

unsafe impl GlobalAlloc for SimpleAlloc {
	unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
		let head = self.head.get();

		let align = layout.align();
		let res = unsafe { *head % align };
		let start = unsafe { if res == 0 { *head } else { *head + align - res } };
		if start + align > self.end {
			return ptr::null_mut();
		} else {
			unsafe { *head = start + align; }
			return start as *mut u8;
		}
	}

	unsafe fn dealloc(&self, _: *mut u8, _: Layout) {
		// not implemented
	}
}

#[alloc_error_handler]
fn on_oom(layout: _Layout) -> ! {
	loop {}
}
