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

#ifndef _ORDERED_MAP_HPP
#define _ORDERED_MAP_HPP

#include <unordered_map>
#include <functional>
#include <memory>
#include <utility>

template<
	class Key,
    class T,
    class Hash = std::hash<Key>,
    class KeyEqual = std::equal_to<Key>,
    class Allocator = std::allocator<std::pair<const Key, T>>
> class ordered_map {
  public:

    using unordered_map_t = std::unordered_map<Key, T, Hash, KeyEqual, Allocator>;
    using vector_t = std::vector<Key>;

	auto reserve(size_t size){
		order.reserve(size);
		return internal_map.reserve(size);
	}

	template<class... Args>
	decltype(auto) emplace(Args&&... args) {
		const auto first = std::get<0>(std::forward_as_tuple(args...));
		order.emplace_back(first);
		return internal_map.emplace(std::forward<Args>(args)...);
	}

	size_t size(){
		return internal_map.size();
	}

	const T& operator[](const std::string& index) const {
        return internal_map[index];
    }

	// This is needed so we can use the container in an iterator context like ranged fors.
    template<class _map_t, class _vec_t>
    class ordered_map_iterator_t {
		using unordered_map_iter_t = typename _map_t::iterator;
    	using vec_iter_t = typename _vec_t::iterator;

		_map_t& map;
		_vec_t& vec;

        unordered_map_iter_t map_iter;
        vec_iter_t vec_iter;

	  public:

    	using reference = typename unordered_map_iter_t::reference;
        using difference_type = typename unordered_map_iter_t::difference_type;
        using value_type = typename unordered_map_iter_t::value_type;
        using pointer = typename unordered_map_iter_t::reference;
        using iterator_category = typename unordered_map_iter_t::iterator_category;

        ordered_map_iterator_t(_map_t& map,
                 _vec_t& vec) : map(map), vec(vec){
        }

        ordered_map_iterator_t begin() {
            vec_iter = vec.begin();
            map_iter = map.find(*vec_iter);
            return *this;
        }

        ordered_map_iterator_t end() {
            vec_iter = vec.end();
            map_iter = map.find(*(vec_iter - 1));
            return *this;
        }

        ordered_map_iterator_t operator ++(){
            auto tmp = ++vec_iter;
            tmp = (tmp == vec.end()? --tmp: tmp);
            map_iter = map.find(*tmp);
            return *this;
        }

        bool operator ==(const ordered_map_iterator_t& rhs) const {
            return vec_iter == rhs.vec_iter;
        }

        bool operator !=(const ordered_map_iterator_t& rhs) const {
            return vec_iter != rhs.vec_iter;
        }

        reference operator *() const {
            return *map_iter;
        }
    };

    using iterator = ordered_map_iterator_t<unordered_map_t, vector_t>;
    using const_iterator = ordered_map_iterator_t<unordered_map_t, vector_t>;

    iterator it{internal_map, order};

    const_iterator begin() {
        return it.begin();
    }

    const_iterator end() {
        return it.end();
    }

  private:
    unordered_map_t internal_map;
	vector_t order;

};

#endif //_ORDERED_MAP_HPP