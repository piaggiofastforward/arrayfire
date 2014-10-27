#include <af/image.h>
#include "error.hpp"

namespace af
{

array bilateral(const array &in, const float spatial_sigma, const float chromatic_sigma, bool is_color)
{
    af_array out = 0;
    AF_THROW(af_bilateral(&out, in.get(), spatial_sigma, chromatic_sigma, is_color));
    return array(out);
}

}
