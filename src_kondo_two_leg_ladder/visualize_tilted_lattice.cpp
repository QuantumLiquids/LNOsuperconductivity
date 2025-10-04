#include "tilted_zigzag_lattice.h"

#include <cstdlib>
#include <iostream>
#include <string>

int main(int argc, char **argv) {
  std::size_t Ly = 4;
  std::size_t Lx = 12;
  std::string output_path = "tilted_lattice_test.svg";
  bool include_pbc = true;

  if (argc >= 3) {
    Ly = static_cast<std::size_t>(std::strtoul(argv[1], nullptr, 10));
    Lx = static_cast<std::size_t>(std::strtoul(argv[2], nullptr, 10));
  }
  if (argc >= 4) {
    output_path = argv[3];
  }
  if (argc >= 5) {
    include_pbc = std::strtol(argv[4], nullptr, 10) != 0;
  }

  TiltedZigZagLattice lattice(Ly, Lx);
  lattice.DumpSVG(output_path, 80.0, include_pbc);

  std::cout << "Generated tilted lattice SVG at " << output_path
            << " with Ly=" << Ly << " and Lx=" << Lx << std::endl;
  if (include_pbc) {
    std::cout << "PBC wrap connections are drawn in blue." << std::endl;
  }
  return 0;
}
