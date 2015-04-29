// minimal SVD implementation for calculating feature points from hermite data
// works in C++ and GLSL

// public domain

#ifndef USE_GLSL
#define USE_GLSL 0
#endif

#ifndef USE_WEBGL
#define USE_WEBGL 0
#endif

#ifndef DEBUG_SVD
#define DEBUG_SVD (!USE_GLSL)
#endif

#ifndef SVD_COMPARE_REFERENCE
#define SVD_COMPARE_REFERENCE 1
#endif

#define SVD_NUM_SWEEPS 5

#if USE_GLSL

// GLSL prerequisites

#define IN(t,x) in t x
#define OUT(t, x) out t x
#define INOUT(t, x) inout t x
#define rsqrt inversesqrt

#define SWIZZLE_XYZ(v) v.xyz

#else

// C++ prerequisites

#include <math.h>
#include <glm/glm.hpp>
#include <glm/gtc/swizzle.hpp>
using namespace glm;

#define abs fabs
#define sqrt sqrtf
#define max(x,y) (((x)>(y))?(x):(y))
#define IN(t,x) t x
#define OUT(t, x) t &x
#define INOUT(t, x) t &x

float rsqrt(float x) {
    return 1.0 / sqrt(x);
}

#define SWIZZLE_XYZ(v) vec3(swizzle<X,Y,Z>(v))

#endif

#if DEBUG_SVD

// Debugging
////////////////////////////////////////////////////////////////////////////////

void dump_vec3(vec3 v) {
    printf("(%.5f %.5f %.5f)\n", v[0], v[1], v[2]);
}

void dump_vec4(vec4 v) {
    printf("(%.5f %.5f %.5f %.5f)\n", v[0], v[1], v[2], v[3]);
}

void dump_mat3(mat3 m) {
    printf("(%.5f %.5f %.5f\n %.5f %.5f %.5f\n %.5f %.5f %.5f)\n",
        m[0][0], m[0][1], m[0][2],
        m[1][0], m[1][1], m[1][2],
        m[2][0], m[2][1], m[2][2]);
}

#endif

// SVD
////////////////////////////////////////////////////////////////////////////////

const float Tiny_Number = 1.e-20;

void givens_coeffs_sym(float a_pp, float a_pq, float a_qq, OUT(float,c), OUT(float,s)) {
    if (a_pq == 0.0) {
        c = 1.0;
        s = 0.0;
        return;
    }
    float tau = (a_qq - a_pp) / (2.0 * a_pq);
    float stt = sqrt(1.0 + tau * tau);
    float tan = 1.0 / ((tau >= 0.0) ? (tau + stt) : (tau - stt));
    c = rsqrt(1.0 + tan * tan);
    s = tan * c;
}

void svd_rotate_xy(INOUT(float,x), INOUT(float,y), IN(float,c), IN(float,s)) {
    float u = x; float v = y;
    x = c * u - s * v;
    y = s * u + c * v;
}

void svd_rotateq_xy(INOUT(float,x), INOUT(float,y), INOUT(float,a), IN(float,c), IN(float,s)) {
    float cc = c * c; float ss = s * s;
    float mx = 2.0 * c * s * a;
    float u = x; float v = y;
    x = cc * u - mx + ss * v;
    y = ss * u + mx + cc * v;
}

#if USE_WEBGL

void svd_rotate01(INOUT(mat3,vtav), INOUT(mat3,v)) {
    if (vtav[0][1] == 0.0) return;
    
    float c, s;
    givens_coeffs_sym(vtav[0][0], vtav[0][1], vtav[1][1], c, s);
    svd_rotateq_xy(vtav[0][0],vtav[1][1],vtav[0][1],c,s);
    svd_rotate_xy(vtav[0][2], vtav[1][2], c, s);
    vtav[0][1] = 0.0;
    
    svd_rotate_xy(v[0][0], v[0][1], c, s);
    svd_rotate_xy(v[1][0], v[1][1], c, s);
    svd_rotate_xy(v[2][0], v[2][1], c, s);
}

void svd_rotate02(INOUT(mat3,vtav), INOUT(mat3,v)) {
    if (vtav[0][2] == 0.0) return;
    
    float c, s;
    givens_coeffs_sym(vtav[0][0], vtav[0][2], vtav[2][2], c, s);
    svd_rotateq_xy(vtav[0][0],vtav[2][2],vtav[0][2],c,s);
    svd_rotate_xy(vtav[0][1], vtav[1][2], c, s);
    vtav[0][2] = 0.0;
    
    svd_rotate_xy(v[0][0], v[0][2], c, s);
    svd_rotate_xy(v[1][0], v[1][2], c, s);
    svd_rotate_xy(v[2][0], v[2][2], c, s);
}

void svd_rotate12(INOUT(mat3,vtav), INOUT(mat3,v)) {
    if (vtav[1][2] == 0.0) return;
    
    float c, s;
    givens_coeffs_sym(vtav[1][1], vtav[1][2], vtav[2][2], c, s);
    svd_rotateq_xy(vtav[1][1],vtav[2][2],vtav[1][2],c,s);
    svd_rotate_xy(vtav[0][1], vtav[0][2], c, s);
    vtav[1][2] = 0.0;
    
    svd_rotate_xy(v[0][1], v[0][2], c, s);
    svd_rotate_xy(v[1][1], v[1][2], c, s);
    svd_rotate_xy(v[2][1], v[2][2], c, s);
}

#else

void svd_rotate(INOUT(mat3,vtav), INOUT(mat3,v), IN(int,a), IN(int,b)) {
    if (vtav[a][b] == 0.0) return;
    
    float c, s;
    givens_coeffs_sym(vtav[a][a], vtav[a][b], vtav[b][b], c, s);
    svd_rotateq_xy(vtav[a][a],vtav[b][b],vtav[a][b],c,s);
    svd_rotate_xy(vtav[0][3-b], vtav[1-a][2], c, s);
    vtav[a][b] = 0.0;
    
    svd_rotate_xy(v[0][a], v[0][b], c, s);
    svd_rotate_xy(v[1][a], v[1][b], c, s);
    svd_rotate_xy(v[2][a], v[2][b], c, s);
}

#endif

void svd_solve_sym(IN(mat3,a), OUT(vec3,sigma), INOUT(mat3,v)) {
    // assuming that A is symmetric: can optimize all operations for 
    // the upper right triagonal
    mat3 vtav = a;
    // assuming V is identity: you can also pass a matrix the rotations
    // should be applied to
    // U is not computed
    for (int i = 0; i < SVD_NUM_SWEEPS; ++i) {
        svd_rotate(vtav, v, 0, 1);
        svd_rotate(vtav, v, 0, 2);
        svd_rotate(vtav, v, 1, 2);
    }
    sigma = vec3(vtav[0][0],vtav[1][1],vtav[2][2]);    
}

float svd_invdet(float x, float tol) {
    return (abs(x) < tol || abs(1.0 / x) < tol) ? 0.0 : (1.0 / x);
}

void svd_pseudoinverse(OUT(mat3,o), IN(vec3,sigma), IN(mat3,v)) {
    float d0 = svd_invdet(sigma[0], Tiny_Number);
    float d1 = svd_invdet(sigma[1], Tiny_Number);
    float d2 = svd_invdet(sigma[2], Tiny_Number);
    o = mat3(v[0][0] * d0 * v[0][0] + v[0][1] * d1 * v[0][1] + v[0][2] * d2 * v[0][2],
             v[0][0] * d0 * v[1][0] + v[0][1] * d1 * v[1][1] + v[0][2] * d2 * v[1][2],
             v[0][0] * d0 * v[2][0] + v[0][1] * d1 * v[2][1] + v[0][2] * d2 * v[2][2],
             v[1][0] * d0 * v[0][0] + v[1][1] * d1 * v[0][1] + v[1][2] * d2 * v[0][2],
             v[1][0] * d0 * v[1][0] + v[1][1] * d1 * v[1][1] + v[1][2] * d2 * v[1][2],
             v[1][0] * d0 * v[2][0] + v[1][1] * d1 * v[2][1] + v[1][2] * d2 * v[2][2],
             v[2][0] * d0 * v[0][0] + v[2][1] * d1 * v[0][1] + v[2][2] * d2 * v[0][2],
             v[2][0] * d0 * v[1][0] + v[2][1] * d1 * v[1][1] + v[2][2] * d2 * v[1][2],
             v[2][0] * d0 * v[2][0] + v[2][1] * d1 * v[2][1] + v[2][2] * d2 * v[2][2]);
}

void svd_solve_ATA_ATb(
    IN(mat3,ATA), IN(vec3,ATb), OUT(vec3,x)
) {
    mat3 V = mat3(1.0);
    vec3 sigma;
    
    svd_solve_sym(ATA, sigma, V);
    
    // A = UEV^T; U = A / (E*V^T)
#if DEBUG_SVD
    
    printf("ATA="); dump_mat3(ATA);
    printf("ATb="); dump_vec3(ATb);
    printf("V="); dump_mat3(V);
    printf("sigma="); dump_vec3(sigma);
#endif
    mat3 Vinv;
    svd_pseudoinverse(Vinv, sigma, V);
    x = Vinv * ATb;

#if DEBUG_SVD
    printf("Vinv="); dump_mat3(Vinv);
#endif
}

vec3 svd_vmul_sym(IN(mat3,a), IN(vec3,v)) {
    return vec3(
        dot(a[0],v),
        (a[0][1] * v.x) + (a[1][1] * v.y) + (a[1][2] * v.z),
        (a[0][2] * v.x) + (a[1][2] * v.y) + (a[2][2] * v.z)
    );
}

void svd_mul_ata_sym(OUT(mat3,o), IN(mat3,a))
{
    o[0][0] = a[0][0] * a[0][0] + a[1][0] * a[1][0] + a[2][0] * a[2][0];
    o[0][1] = a[0][0] * a[0][1] + a[1][0] * a[1][1] + a[2][0] * a[2][1];
    o[0][2] = a[0][0] * a[0][2] + a[1][0] * a[1][2] + a[2][0] * a[2][2];
    o[1][1] = a[0][1] * a[0][1] + a[1][1] * a[1][1] + a[2][1] * a[2][1];
    o[1][2] = a[0][1] * a[0][2] + a[1][1] * a[1][2] + a[2][1] * a[2][2];
    o[2][2] = a[0][2] * a[0][2] + a[1][2] * a[1][2] + a[2][2] * a[2][2];
}
    
void svd_solve_Ax_b(IN(mat3,a), IN(vec3,b), OUT(mat3,ATA), OUT(vec3,ATb), OUT(vec3,x)) {
    svd_mul_ata_sym(ATA, a);
    ATb = b * a; // transpose(a) * b;
    svd_solve_ATA_ATb(ATA, ATb, x);
}

// QEF
////////////////////////////////////////////////////////////////////////////////

void qef_add(
    IN(vec3,n), IN(vec3,p),
    INOUT(mat3,ATA), 
    INOUT(vec3,ATb),
    INOUT(vec4,pointaccum))
{
#if DEBUG_SVD
    printf("+plane=");dump_vec4(vec4(n, dot(-p,n)));
#endif    
    ATA[0][0] += n.x * n.x;
    ATA[0][1] += n.x * n.y;
    ATA[0][2] += n.x * n.z;
    ATA[1][1] += n.y * n.y;
    ATA[1][2] += n.y * n.z;
    ATA[2][2] += n.z * n.z;

    float b = dot(p, n);
    ATb += n * b;
    pointaccum += vec4(p,1.0);
}

float qef_calc_error(IN(mat3,A), IN(vec3, x), IN(vec3, b)) {
    vec3 vtmp = b - svd_vmul_sym(A, x);
    return dot(vtmp,vtmp);
}

float qef_solve(
    IN(mat3,ATA), 
    IN(vec3,ATb),
    IN(vec4,pointaccum),
    OUT(vec3,x)
) {
    vec3 masspoint = SWIZZLE_XYZ(pointaccum) / pointaccum.w;
    ATb -= svd_vmul_sym(ATA, masspoint);
    svd_solve_ATA_ATb(ATA, ATb, x);
    float result = qef_calc_error(ATA, x, ATb);
    
    x += masspoint;
        
    return result;
}

// Test
////////////////////////////////////////////////////////////////////////////////

#if USE_GLSL

void main(void) {
}

#else

#if DEBUG_SVD
#undef sqrt
#undef abs
#undef rsqrt
#undef max

#if SVD_COMPARE_REFERENCE
#include "qr_solve.h"
#include "r8lib.h"
#include "qr_solve.c"
#include "r8lib.c"
#endif

int main(void) {
    vec4 pointaccum = vec4(0.0);
    mat3 ATA = mat3(0.0);
    vec3 ATb = vec3(0.0);
    
    #define COUNT 5
    vec3 normals[COUNT] = {
        normalize(vec3( 1.0,1.0,0.0)),
        normalize(vec3( 1.0,1.0,0.0)),
        normalize(vec3(-1.0,1.0,0.0)),
        normalize(vec3(-1.0,2.0,1.0)),
        //normalize(vec3(-1.0,1.0,0.0)),
        normalize(vec3(-1.0,1.0,0.0)),
    };
    vec3 points[COUNT] = {
        vec3(  1.0,0.0,0.3),
        vec3(  0.9,0.1,-0.5),
        vec3( -0.8,0.2,0.6),
        vec3( -1.0,0.0,0.01),
        vec3( -1.1,-0.1,-0.5),
    };
    
    for (int i= 0; i < COUNT; ++i) {
        qef_add(normals[i],points[i],ATA,ATb,pointaccum);
    }
    vec3 com = SWIZZLE_XYZ(pointaccum) / pointaccum.w;
    
    vec3 x;
    float error = qef_solve(ATA,ATb,pointaccum,x);

    printf("masspoint = (%.5f %.5f %.5f)\n", com.x, com.y, com.z);
    printf("point = (%.5f %.5f %.5f)\n", x.x, x.y, x.z);
    printf("error = %.5f\n", error);

#if SVD_COMPARE_REFERENCE
    double a[COUNT*3];
    double b[COUNT];
    
    for (int i = 0; i < COUNT; ++i) {
        b[i] = (points[i].x - com.x)*normals[i].x
             + (points[i].y - com.y)*normals[i].y
             + (points[i].z - com.z)*normals[i].z;
        a[i] = normals[i].x;
        a[i+COUNT] = normals[i].y;
        a[i+2*COUNT] = normals[i].z;
    }
    
    double *c = svd_solve(5,3,a,b,0.1);
    
    vec3 result = com + vec3(c[0], c[1], c[2]);
    r8_free(c);
    printf("reference="); dump_vec3(result);
#endif
    
    return 0;
}
#endif

#endif
