use sprs::linalg::bicgstab;
use sprs::{CsMat, CsVec, TriMat};

use plotters::prelude::*;

use crate::Case::Cartesian;

/* function to ge the interpolation parameters at a given location \xi */
fn interpolation_coefficients(xi: f64, temp: &f64) -> (f64, f64, f64) {
    let alpha = xi / (2.0 - xi);
    let beta = xi / (2.0 - xi);
    let gamma = xi * temp / (2.0 - xi);
    (alpha, beta, gamma)
}

enum Case{
    Cartesian,
    Sphere,
    Cylinder
}

fn main() {
    let nx = 50;
    let dx = 0.1;
    let dt = 0.01;
    let diff = 1.0;
    let temp_r = 1.0;

    let case: Case= Cartesian;


    // cell-centered 1D grid
    let xc: Vec<f64> = (0..nx)
                        .map(|i| dx / 2.0 + i as f64 * dx)
                        .collect();

    println!("Cell centers: {:?}", xc);

    // initial temperature
    let temp_n: Vec<f64> = vec![0.0; nx];

    // fourier coefficients
    let fourier:f64;
    match case {
        Case::Cartesian => fourier = diff * dt / dx / dx,
        Case::Sphere    => fourier = diff * dt / dx / dx,
        Case::Cylinder  => fourier = diff * dt / dx / dx,
    } 
    println!("Fourier no: {}", fourier);

    // discretization coefficients
    let mut awd = vec![fourier; nx];
    let mut aed = vec![fourier; nx];
    let mut b = vec![0.0; nx];

    let mut apd: Vec<f64> = (0..nx)
                            .map(|i| -(1.0 + awd[i] + aed[i]))
                            .collect();

    // apply the bc zero-flux left
    apd[0] = apd[0] + awd[0];

    // apply the Dirichlet to the right
    let (alpha, beta, gamma) = interpolation_coefficients(0.5, &temp_r);
    awd[nx - 1] += beta * aed[nx - 1];
    apd[nx - 1] += alpha * aed[nx - 1];
    b[nx - 1] -= gamma * aed[nx - 1];
    aed[nx - 1] = 0.0;

    // build RHS
    let indices: Vec<usize> = (0..nx).collect();
    let _rhs: Vec<f64> = (0..nx).map(|i| b[i] - temp_n[i]).collect();
    let rhs = CsVec::new(nx, indices, _rhs);

    // build sparse matrix
    let mut triplet = TriMat::new((nx, nx));
    for i in 0..nx {
        // Main Diagonal
        triplet.add_triplet(i, i, apd[i]);

        // West / Subdiagonal (Offset -1)
        if i > 0 {
            triplet.add_triplet(i, i - 1, awd[i]);
        }

        // East / Superdiagonal (Offset +1)
        if i < nx - 1 {
            triplet.add_triplet(i, i + 1, aed[i]);
        }
    }
    let a: CsMat<f64> = triplet.to_csr();

    // build solution vector
    let indices: Vec<usize> = (0..nx).collect();
    let temp_np1 = CsVec::empty(nx);

    match bicgstab::BiCGSTAB::<'_, f64, _, _>::solve(
        a.view(),
        temp_np1.view(),
        rhs.view(),
        1e-8,
        1000,
    ) {
        Ok(_) => println!("Temperature solved!"),
        Err(e) => println!("Solver failed to converge: {:?}", e),
    }

    // plot the results
    let drawing_area = BitMapBackend::new("results/init_temp.png", (600, 400)).into_drawing_area();

    drawing_area.fill(&WHITE).unwrap();

    let mut chart = ChartBuilder::on(&drawing_area)
        .build_cartesian_2d(0..100, 0..100)
        .unwrap();

    chart
        .draw_series(LineSeries::new((0..100)
                                         .map(|x| (x, 100 - x)), &BLACK))
                                         .unwrap();
}
