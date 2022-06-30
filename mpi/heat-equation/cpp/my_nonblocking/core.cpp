// Main solver routines for heat equation solver

#include <mpi.h>

#include "heat.hpp"

// Exchange the boundary values
void exchange(Field& field, const ParallelData parallel)
{
    //     1
    //  4     2
    //     3

    // Send to up, receive from down
    double* sbuf = field.temperature.data(1, 0);
    double* rbuf = field.temperature.data(field.nx+1, 0);
    MPI_Sendrecv(sbuf, field.ny+2, MPI_DOUBLE, parallel.nup,   13,
                 rbuf, field.ny+2, MPI_DOUBLE, parallel.ndown, 13,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Send to down, receive from up
    sbuf = field.temperature.data(field.nx, 0);
    rbuf = field.temperature.data();
    MPI_Sendrecv(sbuf, field.ny+2, MPI_DOUBLE, parallel.ndown, 42,
                 rbuf, field.ny+2, MPI_DOUBLE, parallel.nup,   42,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

// Update the temperature values using five-point stencil */
void evolve(Field& curr, const Field& prev, const double a, const double dt)
{

  // Compilers do not necessarily optimize division to multiplication, so make it explicit
  auto inv_dx2 = 1.0 / (prev.dx * prev.dx);
  auto inv_dy2 = 1.0 / (prev.dy * prev.dy);

  // Determine the temperature field at next time step
  // As we have fixed boundary conditions, the outermost gridpoints
  // are not updated.
  for (int i = 1; i < curr.nx + 1; i++) {
    for (int j = 1; j < curr.ny + 1; j++) {
            curr(i, j) = prev(i, j) + a * dt * (
	       ( prev(i + 1, j) - 2.0 * prev(i, j) + prev(i - 1, j) ) * inv_dx2 +
	       ( prev(i, j + 1) - 2.0 * prev(i, j) + prev(i, j - 1) ) * inv_dy2
               );
    }
  }
}

// Non-blocking comms
void exchange_init(Field& field, ParallelData& parallel) {
    //     1
    //  4     2
    //     3
    // Send to up, receive from down
    double* sbuf_up   = field.temperature.data(1, 0);
    double* rbuf_down = field.temperature.data(field.nx+1, 0);
    MPI_Irecv(rbuf_down, field.ny+2, MPI_DOUBLE, parallel.ndown, 12, MPI_COMM_WORLD, &parallel.requests[0]);
    MPI_Isend(sbuf_up,   field.ny+2, MPI_DOUBLE, parallel.nup,   12, MPI_COMM_WORLD, &parallel.requests[1]);

    // Send to down, receive from up
    double* sbuf_down = field.temperature.data(field.nx, 0);
    double* rbuf_up   = field.temperature.data();
    MPI_Irecv(rbuf_up,   field.ny+2, MPI_DOUBLE, parallel.nup,   12, MPI_COMM_WORLD, &parallel.requests[2]);
    MPI_Isend(sbuf_down, field.ny+2, MPI_DOUBLE, parallel.ndown, 12, MPI_COMM_WORLD, &parallel.requests[3]);
}

void exchange_finalize(ParallelData& parallel) {
    MPI_Waitall(4, parallel.requests, MPI_STATUS_IGNORE);
}

void evolve_interior(Field& curr, const Field& prev, const double a, const double dt) {
    // Compilers do not necessarily optimize division to multiplication, so make it explicit
    auto inv_dx2 = 1.0 / (prev.dx * prev.dx);
    auto inv_dy2 = 1.0 / (prev.dy * prev.dy);

    // Determine the temperature field at next time step
    // As we have fixed boundary conditions, the outermost gridpoints
    // are not updated.
    // for (int i = 1; i < curr.nx + 1; i++) {
    for (int i = 2; i < curr.nx; i++) { // exclude boudaries
        for (int j = 1; j < curr.ny + 1; j++) {
            curr(i, j) = prev(i, j) + a * dt * (
                ( prev(i + 1, j) - 2.0 * prev(i, j) + prev(i - 1, j) ) * inv_dx2 +
                ( prev(i, j + 1) - 2.0 * prev(i, j) + prev(i, j - 1) ) * inv_dy2);
        }
    }
}

void evolve_edges(Field& curr, const Field& prev, const double a, const double dt) {
    // Compilers do not necessarily optimize division to multiplication, so make it explicit
    auto inv_dx2 = 1.0 / (prev.dx * prev.dx);
    auto inv_dy2 = 1.0 / (prev.dy * prev.dy);

    // Determine the temperature field at next time step
    // As we have fixed boundary conditions, the outermost gridpoints
    // are not updated.
    // for (int i = 1; i < curr.nx + 1; i++) {
    for (int i = 1; i < curr.nx+1; i+=curr.nx-1) { // exclude boudaries
      for (int j = 1; j < curr.ny + 1; j++) {
              curr(i, j) = prev(i, j) + a * dt * (
  	       ( prev(i + 1, j) - 2.0 * prev(i, j) + prev(i - 1, j) ) * inv_dx2 +
  	       ( prev(i, j + 1) - 2.0 * prev(i, j) + prev(i, j - 1) ) * inv_dy2);
       }
    }
}
