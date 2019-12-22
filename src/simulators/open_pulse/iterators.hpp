/**
 * This code is part of Qiskit.
 *
 * (C) Copyright IBM 2018, 2019.
 *
 * This code is licensed under the Apache License, Version 2.0. You may
 * obtain a copy of this license in the LICENSE.txt file in the root directory
 * of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Any modifications or derivative works of this code must retain this
 * copyright notice, and modified files need to carry a notice indicating
 * that they have been altered from the originals.
 */

#ifndef _ITERATORS_HPP
#define _ITERATORS_HPP

template <typename T>
struct iterator_extractor { using type = typename T::iterator; };

template <typename T>
struct iterator_extractor<T const> { using type = typename T::const_iterator; };

/**
 * Python-like `enumerate()` for C++14 ranged-for
 *
 * I wish I'd had this included in the STL :)
 *
 * Usage:
 * ```c++
 * for(auto& elem: index(vec)){
 *     std::cout << "Index: " << elem.first << " Element: " << elem.second;
 * }
 * ```
 **/
template <typename T>
class Indexer {
public:
    class _Iterator {
        using inner_iterator =  typename iterator_extractor<T>::type;
        using inner_reference = typename std::iterator_traits<inner_iterator>::reference;
    public:
        using reference = std::pair<size_t, inner_reference>;

        _Iterator(inner_iterator it): _pos(0), _it(it) {}

        reference operator*() const {
            return reference(_pos, *_it);
        }

        _Iterator& operator++() {
            ++_pos;
            ++_it;
            return *this;
        }

        _Iterator operator++(int) {
            _Iterator tmp(*this);
            ++*this;
            return tmp;
        }

        bool operator==(_Iterator const& it) const {
            return _it == it._it;
        }
        bool operator!=(_Iterator const& it) const {
            return !(*this == it);
        }

    private:
        size_t _pos;
        inner_iterator _it;
    };

    Indexer(T& t): _container(t) {}

    _Iterator begin() const {
        return _Iterator(_container.begin());
    }
    _Iterator end() const {
        return _Iterator(_container.end());
    }

private:
    T& _container;
}; // class Indexer

template <typename T>
Indexer<T> enumerate(T& t) { return Indexer<T>(t); }

#endif