#ifndef LOAD_DATA
#define LOAD_DATA

#include <fstream>
#include <iostream>
#include <vector>

std::vector<std::vector<double>> read_csv(const std::string &filename,
                                          std::size_t input_size);

#endif
