# Create a library called "SiFeCalo" which includes the source files given in
# the array . The extension is already found.  Any number of sources could be
# listed here.

set(INCLUDE_DIRECTORIES
    ${BASE_INCLUDE_DIRECTORIES} ${CMAKE_SOURCE_DIR}/shipdata
    ${CMAKE_SOURCE_DIR}/SND/SiFeCalo ${VMC_INCLUDE_DIRS} ${genfit2_INCDIR})

include_directories(${INCLUDE_DIRECTORIES} ${FAIRROOT_INCLUDE_DIR})
include_directories(SYSTEM ${SYSTEM_INCLUDE_DIRECTORIES})

set(LINK_DIRECTORIES ${ROOT_LIBRARY_DIR} ${FAIRROOT_LIBRARY_DIR}
                     ${genfit2_LIBDIR})

link_directories(${LINK_DIRECTORIES})

set(SRCS SiFeCalo.cxx SiFeCaloPoint.cxx SiFeCaloHit.cxx)

set(HEADERS)
set(LINKDEF SiFeCaloLinkDef.h)
set(LIBRARY_NAME SiFeCalo)
set(DEPENDENCIES
    Base
    ShipData
    GeoBase
    ParBase
    Geom
    Core
    FairLogger::FairLogger)

generate_library()
