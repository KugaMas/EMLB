#include "edn.hpp"
#include <array>
#include <vector>
#include <iostream>
#include <pybind11/pybind11.h>

namespace py = pybind11;

void edn::EventDenoisor::initialization(py::array_t<uint64_t> arrts, py::array_t<uint16_t> arrx, py::array_t<uint16_t> arry, py::array_t<bool> arrp) {
    py::buffer_info bufp = arrp.request(), bufx = arrx.request(), bufy = arry.request(), bufts = arrts.request();
    assert(bufx.size == bufy.size && bufy.size == bufp.size && bufp.size == bufts.size);

    evlen = bufts.size;

    ptrts = static_cast<uint64_t *> (bufts.ptr);
    ptrx  = static_cast<uint16_t *> (bufx.ptr);
    ptry  = static_cast<uint16_t *> (bufy.ptr);
    ptrp  = static_cast<bool *> (bufp.ptr);
    
    return;
}

/* Double Window Filter */
py::array_t<uint64_t> edn::DoubleWindowFilter::run(py::array_t<uint64_t> arrts, py::array_t<uint16_t> arrx, py::array_t<uint16_t> arry, py::array_t<bool> arrp) {
    edn::EventDenoisor::initialization(arrts, arrx, arry, arrp);
    
    std::vector<std::array<uint64_t, 4>> ans;
    for(int i = 0; i < evlen; i++) {
        dv::Event event(ptrts[i], ptrx[i], ptry[i], ptrp[i]);

		if(!lastREvents.full()) {
			lastREvents.push_back(event);
			continue;
		}
        
        if (calculateDensity(event) >= numThr) {
            ans.push_back(std::array<uint64_t, 4> {event.ts, event.x, event.y, event.p});
            lastNEvents.push_back(event);
        }
        else if (useDoubleMode) {
            lastNEvents.push_back(event);
        }
    }

    py::array_t<uint64_t> result = py::cast(ans);

    return result;
}

int edn::DoubleWindowFilter::calculateDensity(dv::Event& event) {
    int distance;
    int nCorrelated = 0;
    for (const auto& lastR : lastREvents) {
        distance = std::abs(event.x - lastR.x) + std::abs(event.y - lastR.y);
        if (distance <= disThr) nCorrelated++;
        if (nCorrelated >= numThr) break;
    }

    for (const auto& lastN : lastNEvents) {
        distance = std::abs(event.x - lastN.x) + std::abs(event.y - lastN.y);
        if (distance <= disThr) nCorrelated++;
        if (nCorrelated >= numThr) break;
    }

    return nCorrelated;
}

PYBIND11_MODULE(cdn_utils, m)
{
    m.doc() = "C++ implementation of event denoising algorithm";
    py::class_<edn::DoubleWindowFilter>(m, "dwf")
        .def(py::init<uint16_t, uint16_t, int, int, bool, int>())
        .def("run", &edn::DoubleWindowFilter::run);
}
