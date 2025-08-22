#ifndef HX_MYUTIL_H //hao-xin's myutil
#define HX_MYUTIL_H
#include <cstdlib>

bool IsElectron(size_t i, size_t Ly, size_t Np);
size_t GetNumofMps();
void Show(std::vector<size_t> v);
bool Parser(const int argc, char *argv[],
            size_t &start,
            size_t &end);
bool ParserBondDimension(int argc, char *argv[],
                         std::vector<size_t> &D_set);

bool ParserMeasureSite(const int argc, char *argv[],
                       size_t &start,
                       size_t &end);

#endif //HX_MYUTIL_H