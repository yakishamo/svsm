// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Copyright (c) 2024 SUSE LLC
//
// Author: Roy Hopkins <roy.hopkins@suse.com>

use igvm::registers::{SegmentRegister, X86Register};
use igvm::snp_defs::{SevFeatures, SevSelector, SevVmsa};
use igvm::IgvmDirectiveHeader;
use igvm_defs::IgvmNativeVpContextX64;
use zerocopy07::FromZeroes;

use crate::cmd_options::{Hypervisor, SevExtraFeatures};

pub fn construct_start_context(
    start_rip: u64,
    start_rsp: u64,
    initial_cr3: u64,
) -> Vec<X86Register> {
    let mut vec: Vec<X86Register> = Vec::new();

    // Establish CS as a 32-bit code selector.
    let cs = SegmentRegister {
        attributes: 0xc09b,
        base: 0,
        limit: 0xffffffff,
        selector: 0x08,
    };
    vec.push(X86Register::Cs(cs));

    // Establish all data segments as generic data selectors.
    let ds = SegmentRegister {
        attributes: 0xa093,
        base: 0,
        limit: 0xffffffff,
        selector: 0x10,
    };
    vec.push(X86Register::Ds(ds));
    vec.push(X86Register::Ss(ds));
    vec.push(X86Register::Es(ds));
    vec.push(X86Register::Fs(ds));
    vec.push(X86Register::Gs(ds));

    // CR0.PE | CR0.NE | CR0.ET.
    // CR0.PG is also always included, but will be stripped on platforms that
    // must recalculate the page tables.
    vec.push(X86Register::Cr0(0x80000031));

    vec.push(X86Register::Cr3(initial_cr3));

    // CR4.MCE | CR4.PAE.
    vec.push(X86Register::Cr4(0x60));

    // EFER.LME | EFER.LMA are always included but will be stripped on
    // platforms that need to start with paging disabled.
    vec.push(X86Register::Efer(0x500));

    vec.push(X86Register::Rflags(2));
    vec.push(X86Register::Rip(start_rip));
    vec.push(X86Register::Rsp(start_rsp));

    vec
}

pub fn construct_native_start_context(
    regs: &[X86Register],
    compatibility_mask: u32,
) -> IgvmDirectiveHeader {
    let mut context_box = IgvmNativeVpContextX64::new_box_zeroed();
    let context = context_box.as_mut();

    // Copy values from the register list.
    for reg in regs.iter() {
        match reg {
            X86Register::Rsp(r) => {
                context.rsp = *r;
            }
            X86Register::Rbp(r) => {
                context.rbp = *r;
            }
            X86Register::Rsi(r) => {
                context.rsi = *r;
            }
            X86Register::R8(r) => {
                context.r8 = *r;
            }
            X86Register::R9(r) => {
                context.r9 = *r;
            }
            X86Register::R10(r) => {
                context.r10 = *r;
            }
            X86Register::R11(r) => {
                context.r11 = *r;
            }
            X86Register::R12(r) => {
                context.r12 = *r;
            }
            X86Register::Rip(r) => {
                context.rip = *r;
            }
            X86Register::Rflags(r) => {
                context.rflags = *r;
            }
            X86Register::Idtr(table) => {
                context.idtr_base = table.base;
                context.idtr_limit = table.limit;
            }
            X86Register::Gdtr(table) => {
                context.gdtr_base = table.base;
                context.gdtr_limit = table.limit;
            }
            X86Register::Cs(segment) => {
                context.code_attributes = segment.attributes;
                context.code_base = segment.base.try_into().unwrap();
                context.code_limit = segment.limit;
                context.code_selector = segment.selector;
            }
            X86Register::Ds(segment) => {
                context.data_attributes = segment.attributes;
                context.data_base = segment.base.try_into().unwrap();
                context.data_limit = segment.limit;
                context.data_selector = segment.selector;
            }
            X86Register::Gs(segment) => {
                context.gs_base = segment.base;
            }
            X86Register::Cr0(r) => {
                context.cr0 = *r;
            }
            X86Register::Cr3(r) => {
                context.cr3 = *r;
            }
            X86Register::Cr4(r) => {
                context.cr4 = *r;
            }
            X86Register::Efer(r) => {
                context.efer = *r;
            }
            X86Register::Es(_)
            | X86Register::Fs(_)
            | X86Register::Ss(_)
            | X86Register::Tr(_)
            | X86Register::Pat(_)
            | X86Register::MtrrDefType(_)
            | X86Register::MtrrPhysBase0(_)
            | X86Register::MtrrPhysMask0(_)
            | X86Register::MtrrPhysBase1(_)
            | X86Register::MtrrPhysMask1(_)
            | X86Register::MtrrPhysBase2(_)
            | X86Register::MtrrPhysMask2(_)
            | X86Register::MtrrPhysBase3(_)
            | X86Register::MtrrPhysMask3(_)
            | X86Register::MtrrPhysBase4(_)
            | X86Register::MtrrPhysMask4(_)
            | X86Register::MtrrFix64k00000(_)
            | X86Register::MtrrFix16k80000(_)
            | X86Register::MtrrFix4kE0000(_)
            | X86Register::MtrrFix4kE8000(_)
            | X86Register::MtrrFix4kF0000(_)
            | X86Register::MtrrFix4kF8000(_) => {}
        }
    }

    IgvmDirectiveHeader::X64NativeVpContext {
        compatibility_mask,
        context: context_box,
        vp_index: 0,
    }
}

fn convert_vmsa_segment(segment: &SegmentRegister) -> SevSelector {
    SevSelector {
        base: segment.base,
        limit: segment.limit,
        attrib: (segment.attributes & 0xFF) | ((segment.attributes >> 4) & 0xF00),
        selector: segment.selector,
    }
}

pub fn construct_vmsa(
    regs: &[X86Register],
    gpa_start: u64,
    vtom: u64,
    compatibility_mask: u32,
    extra_features: &Vec<SevExtraFeatures>,
    hypervisor: Hypervisor,
) -> IgvmDirectiveHeader {
    let mut vmsa_box = SevVmsa::new_box_zeroed();
    let vmsa = vmsa_box.as_mut();

    // Copy values from the register list.
    for reg in regs.iter() {
        match reg {
            X86Register::Rsp(r) => {
                vmsa.rsp = *r;
            }
            X86Register::Rbp(r) => {
                vmsa.rbp = *r;
            }
            X86Register::Rsi(r) => {
                vmsa.rsi = *r;
            }
            X86Register::R8(r) => {
                vmsa.r8 = *r;
            }
            X86Register::R9(r) => {
                vmsa.r9 = *r;
            }
            X86Register::R10(r) => {
                vmsa.r10 = *r;
            }
            X86Register::R11(r) => {
                vmsa.r11 = *r;
            }
            X86Register::R12(r) => {
                vmsa.r12 = *r;
            }
            X86Register::Rip(r) => {
                vmsa.rip = *r;
            }
            X86Register::Rflags(r) => {
                vmsa.rflags = *r;
            }
            X86Register::Gdtr(table) => {
                vmsa.gdtr.base = table.base;
                vmsa.gdtr.limit = table.limit as u32;
            }
            X86Register::Idtr(table) => {
                vmsa.idtr.base = table.base;
                vmsa.idtr.limit = table.limit as u32;
            }
            X86Register::Cs(segment) => {
                vmsa.cs = convert_vmsa_segment(segment);
            }
            X86Register::Ds(segment) => {
                vmsa.ds = convert_vmsa_segment(segment);
            }
            X86Register::Es(segment) => {
                vmsa.es = convert_vmsa_segment(segment);
            }
            X86Register::Fs(segment) => {
                vmsa.fs = convert_vmsa_segment(segment);
            }
            X86Register::Gs(segment) => {
                vmsa.gs = convert_vmsa_segment(segment);
            }
            X86Register::Ss(segment) => {
                vmsa.ss = convert_vmsa_segment(segment);
            }
            X86Register::Tr(segment) => {
                vmsa.tr = convert_vmsa_segment(segment);
            }
            X86Register::Cr0(r) => {
                // Remove CR0.PG.
                vmsa.cr0 = *r & !0x8000_0000;
            }
            X86Register::Cr3(r) => {
                vmsa.cr3 = *r;
            }
            X86Register::Cr4(r) => {
                vmsa.cr4 = *r;
            }
            X86Register::Efer(r) => {
                // Remove EFER.LMA and EFER.LME.
                vmsa.efer = *r & !0x500;
            }
            X86Register::Pat(r) => {
                vmsa.pat = *r;
            }
            X86Register::MtrrDefType(_)
            | X86Register::MtrrPhysBase0(_)
            | X86Register::MtrrPhysMask0(_)
            | X86Register::MtrrPhysBase1(_)
            | X86Register::MtrrPhysMask1(_)
            | X86Register::MtrrPhysBase2(_)
            | X86Register::MtrrPhysMask2(_)
            | X86Register::MtrrPhysBase3(_)
            | X86Register::MtrrPhysMask3(_)
            | X86Register::MtrrPhysBase4(_)
            | X86Register::MtrrPhysMask4(_)
            | X86Register::MtrrFix64k00000(_)
            | X86Register::MtrrFix16k80000(_)
            | X86Register::MtrrFix4kE0000(_)
            | X86Register::MtrrFix4kE8000(_)
            | X86Register::MtrrFix4kF0000(_)
            | X86Register::MtrrFix4kF8000(_) => {}
        }
    }

    // Include EFER.SVME on SNP platforms.
    vmsa.efer |= 0x1000;

    // Configure non-zero reset state.
    vmsa.pat = match hypervisor {
        Hypervisor::Vanadium => 0x70106,
        _ => 0x0007040600070406,
    };
    vmsa.xcr0 = 1;

    vmsa.dr6 = match hypervisor {
        Hypervisor::Vanadium => 0xffff0ff0,
        _ => 0,
    };
    vmsa.dr7 = match hypervisor {
        Hypervisor::Vanadium => 0x400,
        _ => 0,
    };

    let mut features = SevFeatures::new();
    features.set_snp(true);
    features.set_restrict_injection(true);
    if vtom != 0 {
        vmsa.virtual_tom = vtom;
        features.set_vtom(true);
    }

    for extra_f in extra_features {
        match extra_f {
            SevExtraFeatures::ReflectVc => features.set_reflect_vc(true),
            SevExtraFeatures::AlternateInjection => features.set_alternate_injection(true),
            SevExtraFeatures::DebugSwap => features.set_debug_swap(true),
            SevExtraFeatures::PreventHostIBS => features.set_prevent_host_ibs(true),
            SevExtraFeatures::SNPBTBIsolation => features.set_snp_btb_isolation(true),
            SevExtraFeatures::VmplSSS => features.set_vmpl_supervisor_shadow_stack(true),
            SevExtraFeatures::SecureTscEn => features.set_secure_tsc(true),
            SevExtraFeatures::VmsaRegProt => features.set_vmsa_reg_protection(true),
            SevExtraFeatures::SmtProtection => features.set_smt_protection(true),
        }
    }

    vmsa.sev_features = features;

    IgvmDirectiveHeader::SnpVpContext {
        gpa: gpa_start,
        compatibility_mask,
        vp_index: 0,
        vmsa: vmsa_box,
    }
}
