# We build the MAM4 box model with CMake since it doesn't have its own build system.
cmake_minimum_required (VERSION 3.24.0)

project(mam4 Fortran C)

include(netcdf.cmake)

set(CMAKE_Fortran_FLAGS "-cpp")
if (CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
  if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -fbacktrace")
  endif()
  set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -Wall -ffree-form -ffree-line-length-0")
  set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -Wno-unused-dummy-argument")
  set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -Wno-unused-variable")

  # This is a hack to make our non-standard inf/nan values work with
  # recent, picky versions of GFortran.
  if (${CMAKE_Fortran_COMPILER_VERSION} GREATER_EQUAL 10)
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -fallow-invalid-boz")
    message(STATUS "BOZ literal hack enabled")
    # we also need this guy
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -fallow-argument-mismatch")
  endif()
elseif (CMAKE_Fortran_COMPILER_ID STREQUAL "Intel")
  if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -traceback")
  endif()
endif()

add_executable(mam4
  e3sm_src/shr_kind_mod.F90
  e3sm_src/shr_const_mod.F90

  box_model_utils/cam_logfile.F90
  box_model_utils/spmd_utils.F90
  box_model_utils/abortutils.F90
  box_model_utils/cam_abortutils.F90
  box_model_utils/ppgrid.F90
  box_model_utils/pmgrid.F90
  box_model_utils/constituents.F90
  box_model_utils/radconstants.F90
  box_model_utils/physconst.F90
  box_model_utils/cam_history.F90
  box_model_utils/units.F90
  box_model_utils/ref_pres.F90
  box_model_utils/dyn_grid.F90
  box_model_utils/buffer.F90
  box_model_utils/physics_buffer.F90
  box_model_utils/physics_types.F90
  box_model_utils/phys_control.F90
  box_model_utils/mo_constants.F90
  box_model_utils/chem_mods.F90
  box_model_utils/mo_tracname.F90
  box_model_utils/mo_chem_utls.F90
  box_model_utils/gffgch.F90
  box_model_utils/wv_saturation.F90
  box_model_utils/error_function.F90
  e3sm_src/shr_spfn_mod.F90
  box_model_utils/infnan.F90
  box_model_utils/seasalt_model.F90
  box_model_utils/time_manager.F90
  box_model_utils/aerodep_flx.F90
  box_model_utils/modal_aero_convproc.F90
  box_model_utils/modal_aero_deposition.F90

  e3sm_src/modal_aero_data.F90

  box_model_utils/rad_constituents.F90

  e3sm_src_modified/modal_aero_wateruptake.F90
  e3sm_src/modal_aero_rename.F90
  box_model_utils/modal_aero_calcsize.F90
  e3sm_src/modal_aero_gasaerexch.F90
  e3sm_src/modal_aero_newnuc.F90
  e3sm_src/modal_aero_coag.F90
  e3sm_src_modified/modal_aero_amicphys.F90
  e3sm_src/modal_aero_initialize_data.F90

  box_model_utils/gaschem_simple.F90
  box_model_utils/cloudchem_simple.F90
  test_drivers/driver.F90
  box_model_utils/main.F90
)
target_link_libraries(mam4 ${NETCDF_LIBS})

# These macros are defined in test_drivers/cambox_config.cpp.in.
target_compile_definitions(mam4 PRIVATE CAMBOX_ACTIVATE_THIS;CAMBOX_DEACTIVATE_THIS;MODAL_AERO;MODAL_AERO_4MODE_MOM;RAIN_EVAP_TO_COARSE_AERO;PCNST=35;PCOLS=1;PVER=1;NBC=1;NPOA=1;NSOA=1)

