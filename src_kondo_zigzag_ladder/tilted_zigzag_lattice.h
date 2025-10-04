#ifndef TILTED_ZIGZAG_LATTICE_H
#define TILTED_ZIGZAG_LATTICE_H

#include <cstddef>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

/**
 * Helper for the 45-degree tilted zig-zag lattice used in the Kondo ladder mapping.
 *
 * - Geometric site ordering follows y + Ly * x so that for Ly=2 we match the legacy
 *   two-leg convention (indices: 0,1 | 2,3 | 4,5 ... along the snake of itinerant+localized).
 * - Each geometric site expands to two MPS sites: even index = itinerant (electron), odd = localized.
 */
class TiltedZigZagLattice {
 public:
  TiltedZigZagLattice(std::size_t ly, std::size_t lx)
      : Ly_(ly), Lx_(lx), num_electrons_(ly * lx) {}

  inline std::size_t ElectronIndex(std::size_t y, std::size_t x) const {
    return 2 * (y + Ly_ * x);
  }

  inline std::size_t LocalizedIndex(std::size_t y, std::size_t x) const {
    return ElectronIndex(y, x) + 1;
  }

  inline std::size_t NumElectrons() const { return num_electrons_; }

  // (y,x) -> (y,x+1)
  std::vector<std::pair<std::size_t, std::size_t>> IntraChainPairs() const {
    std::vector<std::pair<std::size_t, std::size_t>> res;
    if (Lx_ < 2) return res;
    res.reserve(Ly_ * (Lx_ - 1));
    for (std::size_t x = 0; x + 1 < Lx_; ++x) {
      for (std::size_t y = 0; y < Ly_; ++y) {
        res.emplace_back(ElectronIndex(y, x), ElectronIndex(y, x + 1));
      }
    }
    return res;
  }

  // OBC inter-chain couplings for zig-zag chains, these coupling form NN coupling in 2D lattice (t' in our Kondo model)
  std::vector<std::pair<std::size_t, std::size_t>> InterChainNNPairsOBC() const {
    std::vector<std::pair<std::size_t, std::size_t>> res;
    if (Ly_ < 2 || Lx_ < 2) return res;
    for (std::size_t x = 0; x + 1 < Lx_; ++x) {
      const int delta = (x % 2 == 0) ? 1 : -1;
      for (std::size_t y = 0; y < Ly_; ++y) {
        const int target = static_cast<int>(y) + delta;
        if (target >= 0 && target < static_cast<int>(Ly_)) {
          res.emplace_back(ElectronIndex(y, x), ElectronIndex(static_cast<std::size_t>(target), x + 1));
        }
      }
    }
    return res;
  }

  // PBC wrap for inter-chain couplings for zig-zag chains.
  std::vector<std::pair<std::size_t, std::size_t>> InterChainNNPairsPBC() const {
    std::vector<std::pair<std::size_t, std::size_t>> res;
    if (Ly_ < 2 || Lx_ < 2) return res;
    for (std::size_t x = 0; x + 1 < Lx_; ++x) {
      const int delta = (x % 2 == 0) ? 1 : -1;
      for (std::size_t y = 0; y < Ly_; ++y) {
        const int raw_target = static_cast<int>(y) + delta;
        if (raw_target >= 0 && raw_target < static_cast<int>(Ly_)) {
          continue;  // Already handled by OBC set
        }
        int wrapped = raw_target;
        if (wrapped < 0) {
          wrapped += static_cast<int>(Ly_);
        } else if (wrapped >= static_cast<int>(Ly_)) {
          wrapped -= static_cast<int>(Ly_);
        }
        res.emplace_back(ElectronIndex(y, x), ElectronIndex(static_cast<std::size_t>(wrapped), x + 1));
      }
    }
    return res;
  }

  std::vector<std::size_t> EvenIndicesBetween(std::size_t i, std::size_t j) const {
    std::vector<std::size_t> res;
    if (i == j) return res;
    std::size_t a = i < j ? i : j;
    std::size_t b = i < j ? j : i;
    for (std::size_t k = a + 2; k < b; k += 2) {
      res.push_back(k);
    }
    return res;
  }

  void DumpSVG(const std::string &filepath,
               double scale = 80.0,
               bool include_pbc_wrap = false) const {
    if (num_electrons_ == 0) return;

    const double margin = scale * 0.6;
    std::vector<Coord> coords;
    coords.reserve(num_electrons_);
    double min_x = 1e300;
    double max_x = -1e300;
    double min_y = 1e300;
    double max_y = -1e300;
    for (std::size_t x = 0; x < Lx_; ++x) {
      for (std::size_t y = 0; y < Ly_; ++y) {
        Coord c = ElectronCoord(y, x);
        coords.push_back(c);
        if (c.x < min_x) min_x = c.x;
        if (c.x > max_x) max_x = c.x;
        if (c.y < min_y) min_y = c.y;
        if (c.y > max_y) max_y = c.y;
      }
    }
    if (max_x < min_x || max_y < min_y) return;

    const double width = (max_x - min_x) * scale + 2.0 * margin;
    const double height = (max_y - min_y) * scale + 2.0 * margin;
    const double shift_x = margin - min_x * scale;
    const double shift_y = margin - min_y * scale;

    std::ofstream ofs(filepath);
    if (!ofs.is_open()) return;

    ofs << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << width
        << "\" height=\"" << height << "\">\n";
    ofs << "  <g transform=\"translate(" << shift_x << ',' << shift_y << ")\">\n";

    DrawSegment(ofs, IntraChainPairs(), "#2d6a4f", 2.4, "", scale);
    DrawSegment(ofs, InterChainNNPairsOBC(), "#ff9f1c", 2.0, "6,6", scale);
    if (include_pbc_wrap) {
      DrawSegment(ofs, InterChainNNPairsPBC(), "#1f78b4", 2.0, "4,6", scale);
    }

    for (std::size_t i = 0; i < coords.size(); ++i) {
      const auto &c = coords[i];
      const double px = c.x * scale;
      const double py = c.y * scale;
      ofs << "    <circle cx=\"" << px << "\" cy=\"" << py
          << "\" r=\"5\" fill=\"#4E79A7\" stroke=\"#1D3557\" stroke-width=\"1\" />\n";
      ofs << "    <text x=\"" << px + 6 << "\" y=\"" << py - 8
          << "\" font-size=\"12\" fill=\"#222\">" << i << "</text>\n";
    }

    ofs << "  </g>\n";
    ofs << "</svg>\n";
  }

 private:
  struct Coord {
    double x;
    double y;
  };

  /**
   * @param y the index of the zig-zag chain
   * @param x the index of the site in the zig-zag chain
   * @return the real space coordinate of the site
   */
  Coord ElectronCoord(std::size_t y, std::size_t x) const {
    Coord c;
    const std::size_t k = x / 2;
    if ((x % 2) == 0) {
      c.x = static_cast<double>(k + y);
      c.y = static_cast<double>(k) - static_cast<double>(y);
    } else {
      c.x = static_cast<double>(k + y);
      c.y = static_cast<double>(k + 1) - static_cast<double>(y);
    }
    return c;
  }

  void DrawSegment(std::ofstream &ofs,
                   const std::vector<std::pair<std::size_t, std::size_t>> &pairs,
                   const std::string &color,
                   double stroke_width,
                   const std::string &dash,
                   double scale) const {
    for (const auto &p : pairs) {
      const auto [y1, x1] = SiteFromElectronIndex(p.first);
      const auto [y2, x2] = SiteFromElectronIndex(p.second);
      const Coord c1 = ElectronCoord(y1, x1);
      const Coord c2 = ElectronCoord(y2, x2);
      ofs << "    <line x1=\"" << c1.x * scale << "\" y1=\"" << c1.y * scale
          << "\" x2=\"" << c2.x * scale << "\" y2=\"" << c2.y * scale
          << "\" stroke=\"" << color << "\" stroke-width=\"" << stroke_width << "\"";
      if (!dash.empty()) {
        ofs << " stroke-dasharray=\"" << dash << "\"";
      }
      ofs << " />\n";
    }
  }

  std::pair<std::size_t, std::size_t> SiteFromElectronIndex(std::size_t electron_index) const {
    const std::size_t s = electron_index / 2;
    return {s % Ly_, s / Ly_};
  }

  const std::size_t Ly_;
  const std::size_t Lx_;
  const std::size_t num_electrons_;
};

#endif  // TILTED_ZIGZAG_LATTICE_H


