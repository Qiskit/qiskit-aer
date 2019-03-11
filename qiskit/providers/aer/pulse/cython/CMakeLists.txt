find_package(Cython REQUIRED)

# Frame change

add_cython_target(frame_change frame_change.pyx CXX)
