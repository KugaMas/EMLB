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
        void initialization(py::array_t<uint64_t> arrts, py::array_t<uint16_t> arrx, py::array_t<uint16_t> arry, py::array_t<bool> arrp);
    };

    class DoubleWindowFilter: public EventDenoisor {
    private:
        int mLen;
        int numThr;
        int disThr;
        bool useDoubleMode;
        boost::circular_buffer<dv::Event> lastREvents;
        boost::circular_buffer<dv::Event> lastNEvents;
    public:
        DoubleWindowFilter(uint16_t sizeX, uint16_t sizeY, int numThr, int disThr, bool useDoubleMode, int mLen) : EventDenoisor(sizeX, sizeY) {
            this->numThr = numThr;
            this->disThr = disThr;
            this->useDoubleMode = useDoubleMode;
            this->mLen = useDoubleMode ? mLen : mLen / 2;   
            this->lastREvents.set_capacity(mLen);
            this->lastNEvents.set_capacity(mLen);
        };
        py::array_t<uint64_t> run(py::array_t<uint64_t> arrts, py::array_t<uint16_t> arrx, py::array_t<uint16_t> arry, py::array_t<bool> arrp);

        // Addtional function
        int calculateDensity(dv::Event& event);
    };

    // class MultiLayerPerceptronFilter : public EventDenoisor {
    // private:
    // public:
    //     DoubleWindowFilter(uint16_t sizeX, uint16_t sizeY, int numThr, int disThr, bool useDoubleMode, int mLen) : EventDenoisor(sizeX, sizeY) {
    //         this->numThr = numThr;
    //         this->disThr = disThr;
    //         this->useDoubleMode = useDoubleMode;
    //         this->mLen = useDoubleMode ? mLen : mLen / 2;   
    //         this->lastREvents.set_capacity(mLen);
    //         this->lastNEvents.set_capacity(mLen);
    //     };
    // }
}

#endif