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
    class iterator {

		using map_iter_t = typename std::unordered_map<Key, T, Hash, KeyEqual, Allocator>::iterator;
    	using vector_iter_t = typename std::vector<Key>::iterator;

		std::unordered_map<Key, T, Hash, KeyEqual, Allocator>& map;
		std::vector<Key>& vec;

        map_iter_t map_iter;
        vector_iter_t vec_iter;

	  public:

    	using reference = typename map_iter_t::reference;

        iterator(std::unordered_map<std::string, int>& map,
               std::vector<std::string>& vec) : map(map), vec(vec){
        }


        iterator begin(){
            vec_iter = vec.begin();
            map_iter = map.find(*vec_iter);
            return *this;
        }

        iterator end(){
            vec_iter = vec.end();
            map_iter = map.end();
            return *this;
        }

        iterator operator ++(){
            vec_iter++;
            map_iter = map.find(*vec_iter);
            return *this;
        }

        bool operator !=(const iterator& rhs){
            return vec_iter != rhs.vec_iter;
        }

        T& operator *(){
            return map[*vec_iter];
        }
    };

    iterator it{internal_map, order};

    iterator begin(){
        return it.begin();
    }

    iterator end(){
        return it.end();
    }


  private:
    std::unordered_map<Key, T, Hash, KeyEqual, Allocator> internal_map;
	std::vector<Key> order;

};

#endif //_ORDERED_MAP_HPP