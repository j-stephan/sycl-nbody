/* Copyright (c) 2019, Jan Stephan
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms of the BSD
 * license. See the LICENSE file for details.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <future>
#include <iostream>
#include <iterator>
#include <random>
#include <string>
#include <vector>
#include <utility>

#define SYCL_SIMPLE_SWIZZLES
#include <CL/sycl.hpp>

constexpr auto eps = 0.001f;
constexpr auto eps2 = eps * eps;
constexpr auto damping = 0.5f;
constexpr auto delta_time = 0.2f;
constexpr auto iterations = 10;

using float3 = cl::sycl::float3;
using float4 = cl::sycl::float4;

/******************************************************************************
 * Device-side N-Body
 *****************************************************************************/
auto force_calculation(float4 body_pos,

                         cl::sycl::accessor<float4, 1,
                            cl::sycl::access::mode::read,
                            cl::sycl::access::target::global_buffer> positions,

                         std::size_t n)
-> float3
{
    auto acc = float3{0.f, 0.f, 0.f};

    for(auto i = 0ul; i < n; ++i)
    {
        // r_ij [3 FLOPS]
        const auto r = positions[i].xyz() - body_pos.xyz();

        // dist_sqr = dot(r_ij, r_ij) + EPS^2 [6 FLOPS]
        const auto dist_sqr = cl::sycl::fma(r.x(), r.x(), cl::sycl::fma(
                                                r.y(), r.y(), cl::sycl::fma(
                                                    r.z(), r.z(), eps2)));

        // inv_dist_cube = 1/dist_sqr^(3/2) [4 FLOPS]
        auto dist_sixth = dist_sqr * dist_sqr * dist_sqr;
        auto inv_dist_cube = cl::sycl::rsqrt(dist_sixth);

        // s = m_j * inv_dist_cube [1 FLOP]
        const auto s = float{positions[i].w()} * inv_dist_cube;
        const auto s3 = float3{s, s, s};

        // a_i = a_i + s * r_ij [6 FLOPS]
        acc = cl::sycl::fma(r, s3, acc);
    }

    return acc;
}

struct body_integrator
{  
    cl::sycl::accessor<float4, 1, cl::sycl::access::mode::read,
                       cl::sycl::access::target::global_buffer> old_pos;

    cl::sycl::accessor<float4, 1, cl::sycl::access::mode::discard_write,
                       cl::sycl::access::target::global_buffer> new_pos;
    
    cl::sycl::accessor<float4, 1, cl::sycl::access::mode::read_write,
                       cl::sycl::access::target::global_buffer> vel;

    std::size_t n;  // bodies

    auto operator()() -> void
    {
        for(auto i = 0ul; i < n; ++i)
        {
            auto position = old_pos[index];
            auto accel = force_calculation_d(position, old_pos, n);

            /*
             * acceleration = force / mass
             * new velocity = old velocity + acceleration * delta_time
             * note that the body's mass is canceled out here and in
             *  body_body_interaction. Thus force == acceleration
             */
            auto velocity = vel[i];

            velocity.xyz() += accel.xyz() * delta_time;
            velocity.xyz() *= damping;

            position.xyz() += velocity.xyz() * delta_time; 

            new_pos[i] = position;
            vel[i] = velocity;
        }
    }
};

struct xilinx_selector
{
    auto operator()(const cl::sycl::device& dev) const -> int
    {
        const auto vendor = dev.get_info<cl::sycl::info::device::vendor>();
        return (vendor.find("Xilinx") != std::string::npos) ? 1 : -1;
    }
};

auto main() -> int
{
    try
    {
        // --------------------------------------------------------------------
        // init SYCL
        // --------------------------------------------------------------------

        // create queue on device
        auto exception_handler = [] (cl::sycl::exception_list exceptions)
        {
            for(std::exception_ptr e : exceptions)
            {
                try
                {
                    std::rethrow_exception(e);
                }
                catch(const cl::sycl::exception& err)
                {                    
                    std::cerr << "Caught asynchronous SYCL exception: "
                              << err.what() << std::endl;
                }
            }
        };
    
        auto queue = cl::sycl::queue{xilinx_selector{}, exception_handler,
                        cl::sycl::property::queue::enable_profiling{}};


        // --------------------------------------------------------------------
        // Init host memory
        // --------------------------------------------------------------------
        constexpr auto n = 262144ul;

        auto old_positions = std::vector<float4>{};
        auto new_positions = std::vector<float4>{};
        auto velocities = std::vector<float4>{};

        old_positions.resize(n);
        new_positions.resize(n);
        velocities.resize(n);

        auto gen = std::mt19937{std::random_device{}()};
        auto dis = std::uniform_real_distribution<float>{-42.f, 42.f};
        auto init_vec = [&]()
        {
            return float4{dis(gen), dis(gen), dis(gen), 0.f};
        };

        std::generate(begin(old_positions), end(old_positions), init_vec);
        std::fill(begin(new_positions), end(new_positions), float4{});
        std::generate(begin(velocities), end(velocities), init_vec);

        // --------------------------------------------------------------------
        // Init device memory
        // --------------------------------------------------------------------
        auto d_old_positions = cl::sycl::buffer<float4, 1>{
            cl::sycl::range<1>{old_positions.size()}};
        d_old_positions.set_write_back(false);

        auto d_new_positions = cl::sycl::buffer<float4, 1>{
            cl::sycl::range<1>{new_positions.size()}};
        d_new_positions.set_write_back(false);

        auto d_velocities = cl::sycl::buffer<float4, 1>{
            cl::sycl::range<1>{velocities.size()}};
        d_velocities.set_write_back(false);

        queue.submit([&](cl::sycl::handler& cgh)
        {
            auto acc = d_old_positions.get_access<
                cl::sycl::access::mode::discard_write,
                cl::sycl::access::target::global_buffer>(cgh);

            cgh.copy(old_positions.data(), acc);
        });

        queue.submit([&](cl::sycl::handler& cgh)
        {
            auto acc = d_new_positions.get_access<
                cl::sycl::access::mode::discard_write,
                cl::sycl::access::target::global_buffer>(cgh);

            cgh.copy(new_positions.data(), acc);
        });

        queue.submit([&](cl::sycl::handler& cgh)
        {
            auto acc = d_velocities.get_access<
                cl::sycl::access::mode::discard_write,
                cl::sycl::access::target::global_buffer>(cgh);

            cgh.copy(velocities.data(), acc);
        });


        // --------------------------------------------------------------------
        // execute kernel
        // --------------------------------------------------------------------
        auto first_event = cl::sycl::event{};
        auto last_event = cl::sycl::event{};

        for(auto i = 0; i < iterations; ++i)
        {
            last_event = queue.submit(
            [&, n](cl::sycl::handler& cgh)
            {
                auto old_acc = d_old_positions.get_access<
                    cl::sycl::access::mode::read,
                    cl::sycl::access::target::global_buffer>();

                auto new_acc = d_new_positions.get_access<
                    cl::sycl::access::mode::discard_write,
                    cl::sycl::access::target::global_buffer>();

                auto vel_acc = d_velocities.get_access<
                    cl::sycl::access::mode::read_write,
                    cl::sycl::access::target::global_buffer>();

                auto integrator = body_integrator{old_acc, new_acc, vel_acc, n};
                cgh.single_task(integrator);
            });

            if(i == 0)
                first_event = last_event;

            std::swap(d_old_positions, d_new_positions);
        }
        queue.wait_and_throw();

        // --------------------------------------------------------------------
        // results
        // --------------------------------------------------------------------
        auto start = first_event.get_profiling_info<
            cl::sycl::info::event_profiling::command_start>();
        auto stop = last_event.get_profiling_info<
            cl::sycl::info::event_profiling::command_end>();

        auto time_ns = stop - start;
        auto time_s = time_ns / 1e9;
        auto time_ms = time_ns / 1e6;

        constexpr auto flops_per_interaction = 20.;
        auto interactions = static_cast<double>(n * n);
        auto interactions_per_second = interactions * iterations 
                                       / time_s;
        auto flops = interactions_per_second * flops_per_interaction;
        auto gflops = flops / 1e9;

        std::cout << << n << ";" << time_ms << ";" << gflops << std::endl;
    }
    catch(const cl::sycl::exception& e)
    {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
