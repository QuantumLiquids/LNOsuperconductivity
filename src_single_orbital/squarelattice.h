#ifndef SQUARE_LATTICE_H
#define SQUARE_LATTICE_H

#include <iostream>
#include <cstdlib>
#include <vector>
#include <tuple>
#include <assert.h>

using std::vector;
using std::tuple;
using std::min;
using std::max;

using Link = tuple<size_t, size_t>;

//base class
class DoubleLayerSquareLattice {
 public :
  DoubleLayerSquareLattice(void) = default;

  DoubleLayerSquareLattice(const size_t Ly, const size_t Lx) : Ly(Ly), Lx(Lx), N(2 * Lx * Ly) {}

  inline void Print(void) {
    std::cout << "System size (Lx, Ly) = ( "
              << Lx << ", "
              << Ly << ")" << "\n";
  }

  size_t Ly;
  size_t Lx;
  size_t N; //total site number
  vector<Link> intralayer_links;      //ordered by smaller number site to larger number site
  vector<Link> interlayer_links;
};


/** Square lattice cylinder
 *
 *  example: Ly = 4, Lx = 8, PBC on y-direction and OBC on x-direction
 *
 *     (Ly-1)
 *        3---7--11--15--19--23--27--31(N-1)
 *        |   |   |   |   |   |   |   |
 *        |   |   |   |   |   |   |   |
 *        2---6--10--14--18--22--26--30
 *        |   |   |   |   |   |   |   |
 *        |   |   |   |   |   |   |   |
 *        1---5---9--13--17--21--25--29
 *        |   |   |   |   |   |   |   |
 *        |   |   |   |   |   |   |   |
 *        0---4---8--12---16--20--24--28(N-Ly)
 *
 */

class DoubleLayerSquareCylinder : public DoubleLayerSquareLattice {
 public:
  DoubleLayerSquareCylinder(void) = default;

  DoubleLayerSquareCylinder(const size_t Ly, const size_t Lx);
};

DoubleLayerSquareCylinder::DoubleLayerSquareCylinder(const size_t Ly, const size_t Lx) :
    DoubleLayerSquareLattice(Ly, Lx) {
  assert(Ly > 2);
  //neasert neighbor links
  intralayer_links.reserve(4 * N); //reserve a little more
  for (size_t i = 0; i < N; ++i) {
    const size_t y = i % (2 * Ly); //y coordinate of site i
    const size_t x = i / (2 * Ly); //x coordinate of site i
    const size_t Tx = y + (2 * Ly) * ((x + 1) % Lx); //x-directional translation site of site i
    const size_t Ty = (y + 2) % (2 * Ly) + (2 * Ly) * x;   //y-directional translation site of site i

    intralayer_links.push_back(Link{min(i, Ty), max(i, Ty)});
    if (x < Lx - 1) {
      intralayer_links.push_back(Link{i, Tx});
    }

    if (y % 2 == 0) {
      size_t Tz = i + 1;
      interlayer_links.push_back(Link{i, Tz});
    }
  }
}

class DoubleLayerSquareOBC : public DoubleLayerSquareLattice {
 public:
  DoubleLayerSquareOBC(void) = default;

  DoubleLayerSquareOBC(const size_t Ly, const size_t Lx);
};

DoubleLayerSquareOBC::DoubleLayerSquareOBC(const size_t Ly, const size_t Lx) :
    DoubleLayerSquareLattice(Ly, Lx) {
  //neasert neighbor links
  intralayer_links.reserve(4 * N); //reserve a little more
  interlayer_links.reserve(N / 2);
  for (size_t i = 0; i < N; ++i) {
    const size_t y = i % (2 * Ly); //y coordinate of site i
    const size_t x = i / (2 * Ly); //x coordinate of site i
    const size_t Tx = y + (2 * Ly) * ((x + 1) % Lx); //x-directional translation site of site i
    const size_t Ty = (y + 2) % (2 * Ly) + (2 * Ly) * x;   //y-directional translation site of site i

    if (y < 2 * (Ly - 1)) {
      intralayer_links.push_back(Link{i, Ty});
    }
    if (x < Lx - 1) {
      intralayer_links.push_back(Link{i, Tx});
    }

    if (y % 2 == 0) {
      size_t Tz = i + 1;
      interlayer_links.push_back(Link{i, Tz});
    }
  }
}

/**
 *
 * example: Ly = 4, Lx = 8 , PBC in vertical direction and OBC in horizontal direction
 *
 *     (Ly-1)
 *        3   7  11  15  19  23  27  31(N-1)
 *        \  /\  /\  /\  /\  /\  /\  /\
 *         \/  \/  \/  \/  \/  \/  \/  \
 *         2   6  10  14  18  22  26  30
 *        /\  /\  /\  /\  /\  /\  /\  /
 *       /  \/  \/  \/  \/  \/  \/  \/
 *      1   5   9   13  17  21  25  29
 *      \  /\  /\  /\  /\  /\  /\  /\
 *       \/  \/  \/  \/  \/  \/  \/  \
 *       0   4   8  12   16  20  24  28(N-Ly)
 *
 */
//class SquareRotatedCylinder : public DoubleLayerSquareLattice {
// public:
//  SquareRotatedCylinder(void) = default;
//
//  SquareRotatedCylinder(const size_t Ly, const size_t Lx);
//};
//
//SquareRotatedCylinder::SquareRotatedCylinder(const size_t Ly, const size_t Lx) :
//    DoubleLayerSquareLattice(Ly, Lx) {
//  assert(Ly > 2 && Ly % 2 == 0);
//  //neasert neighbor links
//  nearest_neighbor_links.reserve(2 * N); //reserve a little more
//  for (size_t i = 0; i < N; ++i) {
//    const size_t y = i % Ly; //y coordinate of site i
//    const size_t x = i / Ly; //x coordinate of site i
//    const size_t Ty = (y + 1) % Ly + Ly * x;   //y-directional translation site of site i
//    const size_t Txy = (y + 1) % Ly + Ly * ((x + 1) % Lx); //right-up-directional translation site of site i
//    const size_t Txy2 = (y - 1 + Ly) % Ly + Ly * ((x + 1) % Lx);//right-down-directional translation site of site i
//
//    nearest_neighbor_links.push_back(Link{min(i, Ty), max(i, Ty)});
//
//    if ((x < Lx - 1) && (y % 2 == 0)) {
//      nearest_neighbor_links.push_back(Link{i, Txy});
//      nearest_neighbor_links.push_back(Link{i, Txy2});
//    }
//  }
//
//  next_nearest_neighbor_links.reserve(2 * N);
//  for (size_t i = 0; i < N; ++i) {
//    const size_t y = i % Ly; //y coordinate of site i
//    const size_t x = i / Ly; //x coordinate of site i
//    const size_t Tx = y + Ly * ((x + 1) % Lx); //x-directional translation site of site i
//    const size_t Ty2 = (y + 2) % Ly + Ly * x;   //y-directional translation site of site i
//    if (x < Lx - 1) {
//      next_nearest_neighbor_links.push_back(Link{i, Tx});
//    }
//    next_nearest_neighbor_links.push_back(Link{min(i, Ty2), max(i, Ty2)});//for Ly=4 coincide...
//  }
//
//}
//
/** Torus
 *
 *  example: Ly = 4, Lx = 8, torus
 *
 *     (Ly-1)
 *        3---7--11--15--19--23--27--31(N-1)
 *        |   |   |   |   |   |   |   |
 *        |   |   |   |   |   |   |   |
 *        2---6--10--14--18--22--26--30
 *        |   |   |   |   |   |   |   |
 *        |   |   |   |   |   |   |   |
 *        1---5---9--13--17--21--25--29
 *        |   |   |   |   |   |   |   |
 *        |   |   |   |   |   |   |   |
 *        0---4---8--12---16--20--24--28(N-Ly)
 *
 */
class DoubleLayerSquareTorus : public DoubleLayerSquareLattice {
 public:
  DoubleLayerSquareTorus(void) = default;

  DoubleLayerSquareTorus(const size_t Ly, const size_t Lx);
};

DoubleLayerSquareTorus::DoubleLayerSquareTorus(const size_t Ly, const size_t Lx) :
    DoubleLayerSquareLattice(Ly, Lx) {
  assert(Ly > 2);
  intralayer_links.reserve(4 * N); //reserve a little more
  for (size_t i = 0; i < N; ++i) {
    const size_t y = i % (2 * Ly); //y coordinate of site i
    const size_t x = i / (2 * Ly); //x coordinate of site i
    const size_t Tx = y + (2 * Ly) * ((x + 1) % Lx); //x-directional translation site of site i
    const size_t Ty = (y + 2) % (2 * Ly) + (2 * Ly) * x;   //y-directional translation site of site i

    intralayer_links.push_back(Link{i, Ty});

    intralayer_links.push_back(Link{i, Tx});


    if (y % 2 == 0) {
      size_t Tz = i + 1;
      interlayer_links.push_back(Link{i, Tz});
    }
  }
}

#endif //SQUARE_LATTICE_H