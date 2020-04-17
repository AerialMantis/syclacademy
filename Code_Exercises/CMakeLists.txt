#[[
  SYCL Academy (c)

  SYCL Academy is licensed under a Creative Commons Attribution-ShareAlike 4.0
  International License.

  You should have received a copy of the license along with this work.  If not,
  see <http://creativecommons.org/licenses/by-sa/4.0/>.
]]

# TODO(Gordon): Generalize this to support other implementations
function(add_sycl_executable prefix source)
	add_executable("${prefix}_${source}" "${source}.cpp")
    target_include_directories("${prefix}_${source}" PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/external/catch2)
	target_link_libraries("${prefix}_${source}" PRIVATE Threads::Threads)
    set_target_properties("${prefix}_${source}" PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)

    add_sycl_to_target(
      TARGET "${prefix}_${source}"
      SOURCES "${source}.cpp"
    )
    
    add_test(${source} ${source})
endfunction()

add_subdirectory(Sample)
add_subdirectory(Exercise_1_Getting_Started)