#include "load_data.h"

#include <fstream>
#include <sstream>
#include <vector>

std::vector<std::vector<double>> read_csv(const std::string &filename,
                                          std::size_t input_size) {
  std::ifstream file(filename);
  if (!file)
    throw std::runtime_error("Could not open file: " + filename);

  std::vector<std::vector<double>> rows;
  std::string line;

  while (std::getline(file, line)) {
    if (line.empty())
      continue;

    std::stringstream iss(line);
    std::string cell;
    std::vector<double> lineNumbers;

    while (std::getline(iss, cell, ',')) {
      lineNumbers.push_back(std::stod(cell));
    }

    rows.push_back(lineNumbers);
  }

  return rows;
}
