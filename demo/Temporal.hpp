#pragma once

#include <array>

template <typename T, size_t NumFrames = 2>
class Temporal
{
public:
  using container = std::array<T, NumFrames>;
  using iterator = container::iterator;

  T& getPrevious() { return data[(NumFrames + currentIndex - 1) % NumFrames]; }
  T& getCurrent() { return data[currentIndex]; }

  void proceed() { currentIndex = (currentIndex + 1) % NumFrames; }

  iterator begin() { return data.begin(); }
  iterator end() { return data.end(); }

  constexpr size_t size() const { return NumFrames; }

  T& operator[](size_t idx) { return data[idx]; }

private:
  size_t currentIndex = 0;
  container data;
};
