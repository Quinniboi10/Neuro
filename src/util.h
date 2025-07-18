#pragma once

#include "types.h"

#include <algorithm>
#include <sstream>
#include <cmath>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <fcntl.h>
#include <io.h>
#endif

struct UnicodeTerminalInitializer {
	UnicodeTerminalInitializer() {
#ifdef _WIN32
		SetConsoleOutputCP(CP_UTF8);
#endif
	}
};

static inline UnicodeTerminalInitializer unicodeTerminalInitializer;

// Recursively fills a vector with a given value
template<typename T, typename U>
inline void deepFill(T& dest, const U& val) {
	dest = val;
}

template<typename T, typename U>
inline void deepFill(vector<T>& arr, const U& value) {
	for (auto& element : arr) {
		deepFill(element, value);
	}
}

// Formats a number with commas
inline string formatNum(i64 v) {
	auto s = std::to_string(v);

	int n = s.length() - 3;
	if (v < 0)
		n--;
	while (n > 0) {
		s.insert(n, ",");
		n -= 3;
	}

	return s;
}

// Formats a time
inline string formatTime(u64 timeInMS) {
	long long seconds = timeInMS / 1000;
	long long hours = seconds / 3600;
	seconds %= 3600;
	long long minutes = seconds / 60;
	seconds %= 60;

	string result;

	if (hours > 0)
		result += std::to_string(hours) + "h ";
	if (minutes > 0 || hours > 0)
		result += std::to_string(minutes) + "m ";
	if (seconds > 0 || minutes > 0 || hours > 0)
		result += std::to_string(seconds) + "s";
	if (result == "")
		return std::to_string(timeInMS) + "ms";
	return result;
}