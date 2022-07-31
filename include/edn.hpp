#ifndef EDN_H
#define EDN_H

#include <vector>
#include <iostream>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <boost/circular_buffer.hpp>

namespace py = pybind11;

using namespace std;

namespace dv {

    using Array  = py::array_t<uint64_t>;
    using Vector = std::vector<std::array<uint64_t, 4>>;

    struct Event {
        uint64_t ts;
        uint16_t x;
        uint16_t y;
        bool p;

        Event(uint64_t ts_, uint16_t x_, uint16_t y_, bool p_) : ts(ts_), x(x_), y(y_), p(p_) {}
    }; 
}

namespace edn {
    class EventDenoisor {
    protected:
        int32_t sizeX;
        int32_t sizeY;

        uint32_t evlen;  // Length of noise events

        bool *ptrp;
        uint16_t *ptrx;
        uint16_t *ptry;
        uint64_t *ptrts;

    public:
        EventDenoisor(uint16_t sizeX, uint16_t sizeY) : sizeX(sizeX), sizeY(sizeY) {};
        virtual ~EventDenoisor() {};
        virtual dv::Vector initialization(py::array_t<uint64_t> arrts, py::array_t<uint16_t> arrx, py::array_t<uint16_t> arry, py::array_t<bool> arrp);
    };

    class DoubleWindowFilter: public EventDenoisor {
    private:
        int thres;
        int radius;
        bool useDoubleMode;
        boost::circular_buffer<dv::Event> lastREvents;
        boost::circular_buffer<dv::Event> lastNEvents;
    public:
        DoubleWindowFilter(uint16_t sizeX, uint16_t sizeY, int numMustBeCorrelated, int disThr, bool useDoubleMode, int memSize) : EventDenoisor(sizeX, sizeY) {
            thres = numMustBeCorrelated;
            radius = disThr;
            this->useDoubleMode = useDoubleMode;
            memSize = useDoubleMode ? memSize : memSize / 2;
            lastREvents.set_capacity(memSize);
            lastNEvents.set_capacity(memSize);
        };
        dv::Array run(py::array_t<uint64_t> arrts, py::array_t<uint16_t> arrx, py::array_t<uint16_t> arry, py::array_t<bool> arrp);

        // Addtional function
        int calculateDensity(dv::Event& event);
    };

    class MultiLayerPerceptronFilter: public EventDenoisor {
    private:
        int thres;
        int radius;
        bool usePolarity;
        bool useTimesurface;

        bool *polMap; 
        uint64_t *tsMap;

    public:
        MultiLayerPerceptronFilter(uint16_t sizeX, uint16_t sizeY) : EventDenoisor(sizeX, sizeY) {
            polMap = (bool*)     std::calloc(sizeX * sizeY, sizeof(bool));
            tsMap  = (uint64_t*) std::calloc(sizeX * sizeY, sizeof(uint64_t));
        }
        dv::Array run(py::array_t<uint64_t> arrts, py::array_t<uint16_t> arrx, py::array_t<uint16_t> arry, py::array_t<bool> arrp);
    };

}

#endif