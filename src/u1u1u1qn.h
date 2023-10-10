// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Hao-Xin Wang <wanghaoxin2014@outlook.com>
* Creation Date: 2023-7-1
*
* Description: GraceQ/tensor project. `U(1) \cross U(1) \cross U(1)` QN class.
*/

/**
@file u1u1u1qn.h
@brief `U(1) \cross U(1) \cross U(1)` QN class.
 Avoiding virtual function realization. (except Showable)
 Compatible in reading and writing  QN<U1QNVal, U1QNVal, U1QNVal> data.
*/

#ifndef GQTEN_GQTENSOR_SPECIAL_QN_U1U1U1QN_H
#define GQTEN_GQTENSOR_SPECIAL_QN_U1U1U1QN_H

#include "gqten/framework/vec_hash.h"   //_HASH_XXPRIME_1...
#include "gqten/gqtensor/qn/qn.h"   //QNCardVec
#include "gqten/gqtensor/qn/qnval.h"    //QNVal
#include "gqten/gqtensor/qn/qnval_u1.h" //U1QNVal
#include <boost/serialization/serialization.hpp>

namespace gqten {
namespace special_qn {

class U1U1U1QN : public Showable {
 public:
  U1U1U1QN(void);

  U1U1U1QN(const int val1, const int val2, const int val3);

  U1U1U1QN(const std::string &name1, const int val1,
           const std::string &name2, const int val2,
           const std::string &name3, const int val3
  );

  U1U1U1QN(const U1U1U1QN &);

  //Compatible
  U1U1U1QN(const QNCardVec &qncards);

  U1U1U1QN &operator=(const U1U1U1QN &);

  ~U1U1U1QN(void);

  U1U1U1QN operator-(void) const;

  U1U1U1QN &operator+=(const U1U1U1QN &);

  U1U1U1QN operator+(const U1U1U1QN &rhs) const;

  U1U1U1QN operator-(const U1U1U1QN &rhs) const;

  size_t dim(void) const { return 1; }

  //Compatible
  U1QNVal GetQNVal(const size_t idx) const {
    assert(idx < 4);
    return U1QNVal(vals_[idx]);
  }

  bool operator==(const U1U1U1QN &rhs) const {
    return hash_ == rhs.hash_;
  }

  bool operator!=(const U1U1U1QN &rhs) const {
    return !(*this == rhs);
  }

  //Hashable
  size_t Hash() const { return hash_; }

  void StreamRead(std::istream &);

  void StreamWrite(std::ostream &) const;

  void Show(const size_t indent_level = 0) const override;

 private:
  size_t CalcHash_(void) const;

  int vals_[3];
  size_t hash_;

  friend class boost::serialization::access;

  template<class Archive>
  void serialize(Archive &ar, const unsigned int version) {
    ar & vals_;
    ar & hash_;
  }
};

inline U1U1U1QN::U1U1U1QN(void) : vals_{0, 0, 0, 0}, hash_(CalcHash_()) {}

inline U1U1U1QN::U1U1U1QN(const int val1, const int val2, const int val3) :
    vals_{val1, val2, val3},
    hash_(CalcHash_()) {}

inline U1U1U1QN::U1U1U1QN(const std::string &name1, const int val1,
                          const std::string &name2, const int val2,
                          const std::string &name3, const int val3) :
    vals_{val1, val2, val3}, hash_(CalcHash_()) {}

inline U1U1U1QN::U1U1U1QN(const U1U1U1QN &rhs) : hash_(rhs.hash_) {
  std::copy(std::begin(rhs.vals_), std::end(rhs.vals_), std::begin(vals_));
}

inline U1U1U1QN::U1U1U1QN(const QNCardVec &qncards) {
  assert(qncards.size() == 3);
  for (size_t i = 0; i < 3; i++) {
    const int val = qncards[i].GetValPtr()->GetVal();
    vals_[i] = val;
  }
  hash_ = CalcHash_();
}

inline U1U1U1QN::~U1U1U1QN() {}

inline U1U1U1QN &U1U1U1QN::operator=(const U1U1U1QN &rhs) {
  for (size_t i = 0; i < 3; i++) {
    vals_[i] = rhs.vals_[i];
  }
  hash_ = rhs.hash_;
  return *this;
}

inline U1U1U1QN U1U1U1QN::operator-() const {
  return U1U1U1QN(-vals_[0], -vals_[1], -vals_[2]);
}

inline U1U1U1QN &U1U1U1QN::operator+=(const U1U1U1QN &rhs) {
  for (size_t i = 0; i < 3; i++) {
    vals_[i] += rhs.vals_[i];
  }
  hash_ = CalcHash_();
  return *this;
}

inline U1U1U1QN U1U1U1QN::operator+(const U1U1U1QN &rhs) const {
  return U1U1U1QN(vals_[0] + rhs.vals_[0], vals_[1] + rhs.vals_[1], vals_[2] + rhs.vals_[2]);
}

inline U1U1U1QN U1U1U1QN::operator-(const U1U1U1QN &rhs) const {
  return U1U1U1QN(vals_[0] - rhs.vals_[0], vals_[1] - rhs.vals_[1], vals_[2] - rhs.vals_[2]);
}

inline void U1U1U1QN::StreamRead(std::istream &is) {
  is >> vals_[0];
  is >> vals_[1];
  is >> vals_[2];
  is >> hash_;
//  CalcHash_();
}

inline void U1U1U1QN::StreamWrite(std::ostream &os) const {
  os << vals_[0] << "\n" << vals_[1] << "\n" << vals_[2] << "\n" << hash_ << "\n";
}

inline void U1U1U1QN::Show(const size_t indent_level) const {
  std::cout << IndentPrinter(indent_level)
            << "U1U1U1QN:  ("
            << vals_[0]
            << ", "
            << vals_[1]
            << ", "
            << vals_[2]
            << ")"
            << "\n";
}

inline size_t U1U1U1QN::CalcHash_() const {
  /** a simple realization
   * in 64 bit system size_t has 8 byte = 64 bits.
   * assume -2^15 < u1vals < 2^15, a map is direct
   */
  const size_t segment_const = 1024 * 1024; //2^20
  size_t hash_val1 = vals_[0] + segment_const;
  size_t hash_val2 = vals_[1] + segment_const;
  size_t hash_val3 = vals_[2] + segment_const;
  hash_val2 *= (2 * segment_const);
  hash_val3 *= (2 * segment_const) * (2 * segment_const);
  size_t hash_val = hash_val1 + hash_val2 + hash_val3;
  return ((hash_val << 10) | (hash_val >> 54)); // To avoid collide of QNSector
}

inline std::istream &operator>>(std::istream &is, U1U1U1QN &qn) {
  qn.StreamRead(is);
  return is;
}

inline std::ostream &operator<<(std::ostream &os, const U1U1U1QN &qn) {
  qn.StreamWrite(os);
  return os;
}

inline size_t Hash(const U1U1U1QN &qn) { return qn.Hash(); }

}//special_qn
}//gqten
#endif //GQTEN_GQTENSOR_SPECIAL_QN_U1U1U1QN_H
