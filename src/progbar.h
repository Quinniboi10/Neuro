#pragma once

#include "stopwatch.h"
#include "util.h"

struct ProgressBar {
    Stopwatch<std::chrono::milliseconds> start;

    ProgressBar() {
        start.start();
    }

    inline string report(u64 progress, u64 total, u64 barWidth) {
        std::ostringstream out;

        out << fmt::format("{:>4.0f}% ", static_cast<float>(progress * 100) / total);

        u64 pos = barWidth * progress / total;
        out << "\u2595";
        for (u64 i = 0; i < barWidth - 1; ++i) {
            if (i < pos) out << "\u2588";
            else out << " ";
        }
        out << "\u258F";

        u64 msRemaining = start.elapsed() * total / progress - start.elapsed();

        out << fmt::format(" {}/{} at {:.2f} per sec with {} remaining", progress, total, static_cast<float>(progress) / start.elapsed() * 1000, formatTime(msRemaining));

        return out.str();
    }
};