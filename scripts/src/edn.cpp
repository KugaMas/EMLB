#include "edn.hpp"
#include <array>
#include <vector>
#include <stdlib.h>
#include <iostream>
#include <pybind11/pybind11.h>

namespace py = pybind11;

dv::Vector edn::EventDenoisor::initialization(py::array_t<uint64_t> arrts, py::array_t<uint16_t> arrx, py::array_t<uint16_t> arry, py::array_t<bool> arrp) {
    py::buffer_info bufp = arrp.request(), bufx = arrx.request(), bufy = arry.request(), bufts = arrts.request();
    assert(bufx.size == bufy.size && bufy.size == bufp.size && bufp.size == bufts.size);

    evlen = bufts.size;

    ptrts = static_cast<uint64_t *> (bufts.ptr);
    ptrx  = static_cast<uint16_t *> (bufx.ptr);
    ptry  = static_cast<uint16_t *> (bufy.ptr);
    ptrp  = static_cast<bool *> (bufp.ptr);

    dv::Vector vec;
    vec.reserve(evlen);

    return vec;
}



/* Double Window Filter */
dv::Array edn::DoubleWindowFilter::run(py::array_t<uint64_t> arrts, py::array_t<uint16_t> arrx, py::array_t<uint16_t> arry, py::array_t<bool> arrp) {
    dv::Vector vec = edn::EventDenoisor::initialization(arrts, arrx, arry, arrp);

    for(int i = 0; i < evlen; i++) {
        dv::Event event(ptrts[i], ptrx[i], ptry[i], ptrp[i]);

		if(!lastREvents.full()) {
			lastREvents.push_back(event);
			continue;
		}
        
        if (calculateDensity(event) >= thres) {
            vec.push_back(std::array<uint64_t, 4> {event.ts, event.x, event.y, event.p});
            lastREvents.push_back(event);
        }
        else if (useDoubleMode) {
            lastNEvents.push_back(event);
        }
    }

    dv::Array result = py::cast(vec);

    return result;
}

int edn::DoubleWindowFilter::calculateDensity(dv::Event& event) {
    int distance;
    int nCorrelated = 0;
    for (const auto& lastR : lastREvents) {
        distance = std::abs(event.x - lastR.x) + std::abs(event.y - lastR.y);
        if (distance <= radius) nCorrelated++;
        if (nCorrelated >= thres) break;
    }

    for (const auto& lastN : lastNEvents) {
        distance = std::abs(event.x - lastN.x) + std::abs(event.y - lastN.y);
        if (distance <= radius) nCorrelated++;
        if (nCorrelated >= thres) break;
    }

    return nCorrelated;
}

/* Multi Layer Perceptron Filter */
dv::Array edn::MultiLayerPerceptronFilter::run(py::array_t<uint64_t> arrts, py::array_t<uint16_t> arrx, py::array_t<uint16_t> arry, py::array_t<bool> arrp) {
    dv::Vector vec = edn::EventDenoisor::initialization(arrts, arrx, arry, arrp);

    polMap = (bool*) std::calloc(sizeX * sizeY, sizeof(bool));
    tsMap  = (uint64_t*) std::calloc(sizeX * sizeY, sizeof(uint64_t));

    for(int i = 0; i < evlen; i++) {
        dv::Event event(ptrts[i], ptrx[i], ptry[i], ptrp[i]);
        
        // update matrix
        tsMap[event.x * sizeY + event.y]  = event.ts;
        polMap[event.x * sizeY + event.y] = event.p;

        

    }

    dv::Array result = py::cast(vec);

    return result;
}

PYBIND11_MODULE(cdn_utils, m)
{
    m.doc() = "C++ implementation of event denoising algorithm";
    py::class_<edn::DoubleWindowFilter>(m, "dwf")
        .def(py::init<uint16_t, uint16_t, int, int, bool, int>())
        .def("run", &edn::DoubleWindowFilter::run);

    py::class_<edn::MultiLayerPerceptronFilter>(m, "mlpf")
        .def(py::init<uint16_t, uint16_t>())
        .def("run", &edn::MultiLayerPerceptronFilter::run);
}
