#ifndef SQUARE_LATTICE_H
#define SQUARE_LATTICE_H

#include <iostream>
#include <cstdlib>
#include <vector>
#include <tuple>
#include <cassert>

using std::vector;
using std::tuple;
using std::min;
using std::max;

using Link = tuple<size_t, size_t>;

//base class
class SquareLattice {
 public :
  SquareLattice(void) = default;
  SquareLattice(const size_t Ly, const size_t Lx) : Ly(Ly), Lx(Lx), N(Lx * Ly) {}
  inline void Print(void) {
    std::cout << "System size (Lx, Ly) = ( "
              << Lx << ", "
              << Ly << ")" << "\n";
  }

  size_t Ly;
  size_t Lx;
  size_t N; //total site number
  vector<Link> nearest_neighbor_links;      //ordered by smaller number site to larger number site
  vector<Link> direct_nn_links;
  vector<Link> wind_nn_links;
  vector<Link> next_nearest_neighbor_links; //ordered by smaller number site to larger number site
  vector<Link> direct_nnn_links;
  vector<Link> wind_nnn_links_1;
  vector<Link> wind_nnn_links_2;
};

/** Two-leg ladder
 * 
 * example: Lx = 9, N = 18
 * 
 *      1   3   5   7   9   11  13  15  17
 *      ---------------------------------
 *      |   |   |   |   |   |   |   |   |
 *      |   |   |   |   |   |   |   |   |
 *      ---------------------------------
 *      0   2   4   6   8   10  12  14  16
 */
class SquareLadder : public SquareLattice {
 public:
  SquareLadder(void) = default;
  SquareLadder(const size_t Lx);
};

SquareLadder::SquareLadder(const size_t Lx) : SquareLattice(2, Lx) {
  assert(Lx >= 2);
  // nearest neighbor links
  nearest_neighbor_links.reserve(2 * (Lx - 1) + Lx);
  for (size_t i = 0; i < N; i = i + 2) {
    nearest_neighbor_links.push_back(Link(i, i + 1));
  }
  for (size_t i = 0; i < N - 2; i = i + 2) {
    nearest_neighbor_links.push_back(Link(i, i + 2));
  }
  for (size_t i = 1; i < N - 2; i = i + 2) {
    nearest_neighbor_links.push_back(Link(i, i + 2));
  }

  // next nearest neighbor links
  next_nearest_neighbor_links.reserve(2 * (Lx - 1));
  for (size_t i = 0; i < N - 2; i = i + 2) {
    next_nearest_neighbor_links.push_back(Link(i, i + 3));
  }
  for (size_t i = 1; i < N - 2; i = i + 2) {
    next_nearest_neighbor_links.push_back(Link(i, i + 1));
  }
}

///< 1D chain
class Chain : public SquareLattice {
 public:
  Chain(void) = default;
  Chain(const size_t Lx);
};

Chain::Chain(const size_t Lx) : SquareLattice(1, Lx) {
  nearest_neighbor_links.reserve(N);
  for (size_t i = 0; i < N; ++i) {
    const size_t x = N; //x coordinate of site i
    const size_t Tx = ((x + 1) % Lx); //x-directional translation site of site i

    nearest_neighbor_links.push_back(Link{min(i, Tx), max(i, Tx)});
    if (x < Lx - 1) {
      direct_nn_links.push_back(Link{i, Tx});
    } else {
      wind_nn_links.push_back(Link{Tx, i});
    }
  }
}

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

class SquareCylinder : public SquareLattice {
 public:
  SquareCylinder(void) = default;
  SquareCylinder(const size_t Ly, const size_t Lx);
};

SquareCylinder::SquareCylinder(const size_t Ly, const size_t Lx) :
    SquareLattice(Ly, Lx) {
  assert(Ly > 2 && Ly % 2 == 0);
  //neasert neighbor links
  nearest_neighbor_links.reserve(2 * N); //reserve a little more
  for (size_t i = 0; i < N; ++i) {
    const size_t y = i % Ly; //y coordinate of site i
    const size_t x = i / Ly; //x coordinate of site i
    const size_t Tx = y + Ly * ((x + 1) % Lx); //x-directional translation site of site i
    const size_t Ty = (y + 1) % Ly + Ly * x;   //y-directional translation site of site i

    nearest_neighbor_links.push_back(Link{min(i, Ty), max(i, Ty)});
    if (y < Ly - 1) {
      direct_nn_links.push_back(Link{i, Ty});
    } else {
      wind_nn_links.push_back(Link{Ty, i});
    }
    if (x < Lx - 1) {
      nearest_neighbor_links.push_back(Link{i, Tx});
      direct_nn_links.push_back(Link{i, Tx});
    }
  }

  next_nearest_neighbor_links.reserve(2 * N);
  for (size_t i = 0; i < N; ++i) {
    const size_t y = i % Ly; //y coordinate of site i
    const size_t x = i / Ly; //x coordinate of site i
    const size_t Txy = (y + 1) % Ly + Ly * ((x + 1) % Lx); //right-up-directional translation site of site i
    const size_t Txy2 = (y - 1 + Ly) % Ly + Ly * ((x + 1) % Lx);//right-down-directional translation site of site i
    if (x < Lx - 1) {
      next_nearest_neighbor_links.push_back(Link{i, Txy});
      next_nearest_neighbor_links.push_back(Link{i, Txy2});
      if (y < Ly - 1 && y > 0) {
        direct_nnn_links.push_back(Link{i, Txy});
        direct_nnn_links.push_back(Link{i, Txy2});
      } else if (y == 0) {
        direct_nnn_links.push_back(Link{i, Txy});
        wind_nnn_links_2.push_back(Link{i, Txy2});
      } else { //y == Ly -1
        wind_nnn_links_1.push_back(Link{i, Txy});
        direct_nnn_links.push_back(Link{i, Txy2});
      }
    }
  }
}

class SquareOBC : public SquareLattice {
 public:
  SquareOBC(void) = default;
  SquareOBC(const size_t Ly, const size_t Lx);
};

SquareOBC::SquareOBC(const size_t Ly, const size_t Lx) :
    SquareLattice(Ly, Lx) {
  assert(Ly > 2 && Ly % 2 == 0);
  //neasert neighbor links
  nearest_neighbor_links.reserve(2 * N); //reserve a little more
  for (size_t i = 0; i < N; ++i) {
    const size_t y = i % Ly; //y coordinate of site i
    const size_t x = i / Ly; //x coordinate of site i
    const size_t Tx = y + Ly * ((x + 1) % Lx); //x-directional translation site of site i
    const size_t Ty = (y + 1) % Ly + Ly * x;   //y-directional translation site of site i

    if (y < Ly - 1) {
      nearest_neighbor_links.push_back(Link{i, Ty});
    }
    if (x < Lx - 1) {
      nearest_neighbor_links.push_back(Link{i, Tx});
    }
  }

  next_nearest_neighbor_links.reserve(2 * N);
  for (size_t i = 0; i < N; ++i) {
    const size_t y = i % Ly; //y coordinate of site i
    const size_t x = i / Ly; //x coordinate of site i
    const size_t Txy = (y + 1) % Ly + Ly * ((x + 1) % Lx); //right-up-directional translation site of site i
    const size_t Txy2 = (y - 1 + Ly) % Ly + Ly * ((x + 1) % Lx);//right-down-directional translation site of site i
    if (x < Lx - 1 && y < Ly - 1) {
      next_nearest_neighbor_links.push_back(Link{i, Txy});
    }

    if (x < Lx - 1 && y > 0) {
      next_nearest_neighbor_links.push_back(Link{i, Txy2});
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
class SquareRotatedCylinder : public SquareLattice {
 public:
  SquareRotatedCylinder(void) = default;
  SquareRotatedCylinder(const size_t Ly, const size_t Lx);
};

SquareRotatedCylinder::SquareRotatedCylinder(const size_t Ly, const size_t Lx) :
    SquareLattice(Ly, Lx) {
  assert(Ly > 2 && Ly % 2 == 0);
  //neasert neighbor links
  nearest_neighbor_links.reserve(2 * N); //reserve a little more
  for (size_t i = 0; i < N; ++i) {
    const size_t y = i % Ly; //y coordinate of site i
    const size_t x = i / Ly; //x coordinate of site i
    const size_t Ty = (y + 1) % Ly + Ly * x;   //y-directional translation site of site i
    const size_t Txy = (y + 1) % Ly + Ly * ((x + 1) % Lx); //right-up-directional translation site of site i
    const size_t Txy2 = (y + Ly - 1) % Ly + Ly * ((x + 1) % Lx);//right-down-directional translation site of site i

    nearest_neighbor_links.push_back(Link{min(i, Ty), max(i, Ty)});

    if ((x < Lx - 1) && (y % 2 == 0)) {
      nearest_neighbor_links.push_back(Link{i, Txy});
      nearest_neighbor_links.push_back(Link{i, Txy2});
    }
  }

  next_nearest_neighbor_links.reserve(2 * N);
  for (size_t i = 0; i < N; ++i) {
    const size_t y = i % Ly; //y coordinate of site i
    const size_t x = i / Ly; //x coordinate of site i
    const size_t Tx = y + Ly * ((x + 1) % Lx); //x-directional translation site of site i
    const size_t Ty2 = (y + 2) % Ly + Ly * x;   //y-directional translation site of site i
    if (x < Lx - 1) {
      next_nearest_neighbor_links.push_back(Link{i, Tx});
    }
    next_nearest_neighbor_links.push_back(Link{min(i, Ty2), max(i, Ty2)});//for Ly=4 coincide...
  }

}

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
class SquareTorus : public SquareLattice {
 public:
  SquareTorus(void) = default;
  SquareTorus(const size_t Ly, const size_t Lx);
};

SquareTorus::SquareTorus(const size_t Ly, const size_t Lx) :
    SquareLattice(Ly, Lx) {
  assert(Ly > 2);

  //nearest neighbor links
  nearest_neighbor_links.reserve(3 * N);
  for (size_t i = 0; i < N; ++i) {
    const size_t y = i % Ly; //y coordinate of site i
    const size_t x = i / Ly; //x coordinate of site i
    const size_t Tx = y + Ly * ((x + 1) % Lx); //x-directional translation site of site i
    const size_t Ty = (y + 1) % Ly + Ly * x;   //y-directional translation site of site i

    nearest_neighbor_links.push_back(Link{min(i, Ty), max(i, Ty)});
    nearest_neighbor_links.push_back(Link{min(i, Tx), max(i, Tx)});
  }


  //next nearest neighbor links
  next_nearest_neighbor_links.reserve(3 * N);
  for (size_t i = 0; i < N; ++i) {
    const size_t y = i % Ly; //y coordinate of site i
    const size_t x = i / Ly; //x coordinate of site i
    const size_t Txy = (y + 1) % Ly + Ly * ((x + 1) % Lx); //right-up-directional translation site of site i
    const size_t Txy2 = (y - 1 + Ly) % Ly + Ly * ((x + 1) % Lx);//right-down-directional translation site of site i

    next_nearest_neighbor_links.push_back(Link{min(i, Txy), max(i, Txy)});
    next_nearest_neighbor_links.push_back(Link{min(i, Txy2), max(i, Txy2)});
  }

}

#endif //SQUARE_LATTICE_H