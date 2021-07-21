using Random,Distributions,StaticArrays


function sample_gig(lambda, chi, psi)
    if (chi < 0.0 || psi < 0.0)      || 
       (chi == 0.0 && lambda <= 0.0) ||
       (psi == 0.0 && lambda >= 0.0)
        ArgumentError("invalid parameters for GIG")
    end

    if chi < eps(Float64)*10
        if lambda > 0.0
            return rand(Gamma(lambda, psi/2))
        else
            return 1/rand(Gamma(-lambda,psi/2))
        end
    elseif psi < eps(Float64)*10
        if lambda > 0.0
            return 1/rand(Gamma(lambda,chi/2))
        else
            return rand(Gamma(-lambda,chi/2))
        end
    else
        lambda_old = lambda
        if lambda < 0.0 
            lambda = -lambda
        end
        alpha = sqrt(chi/psi)
        omega = sqrt(psi*chi)
        if lambda > 2.0 || omega > 3.0
            return gig_ROU_shift(lambda, lambda_old, omega, alpha)
        elseif lambda >= 1.0-2.25*(omega^2) || omega > 0.2
            return gig_ROU_noshift(lambda, lambda_old, omega, alpha)
        elseif lambda >= 0.0 && omega > 0.0
            return gig_concave(lambda, lambda_old, omega, alpha)
        end
    end
end

function gig_ROU_shift(lambda, lambda_old, omega, alpha)
    t = 0.5 * (lambda-1.0)
    s = 0.25 * omega

    xm = gig_mode(lambda,omega)
    nc = t*log(xm) - s*(xm + 1.0/xm)

    a = -(2.0 * (lambda+1.0)/omega + xm)
    b = (2.0 * (lambda-1.0)*xm/omega - 1)
    c = xm

    p = b - a^2/3.0
    q = 2.0*a^3/27.0 - a*b/3.0 + c

    fi = acos(-q/(2.0*sqrt(-p^3/27.0)))
    fak = 2.0*sqrt(-p/3)
    y1 = fak * cos(fi/3.0) - a/3.0
    y2 = fak * cos(fi/3.0 + 4.0/3.0*pi) - a/3.0

    uplus = (y1 - xm) * exp(t*log(y1) - s*(y1 + 1.0/y1) - nc)
    uminus = (y2 - xm) * exp(t*log(y2) - s*(y2 + 1.0/y2) - nc)

    while true
        U = uminus + rand(Uniform()) * (uplus - uminus)
        V = rand(Uniform())
        X = U/V + xm
        if X > 0.0 && log(V) <= t*log(X) - s*(X + 1.0/X) - nc
            if lambda_old < 0.0
                return alpha/X
            else
                return alpha*X
            end
        end
    end
end

function gig_ROU_noshift(lambda, lambda_old, omega, alpha)
    t = 0.5 * (lambda - 1.0)
    s = 0.25 * omega
    xm = gig_mode(lambda,omega)
    nc = t*log(xm) - s*(xm + 1.0/xm)
    ym = ((lambda+1.0) + sqrt((lambda + 1.0)^2 + omega^2))/omega
    um = exp(0.5*(lambda+1.0)*log(ym) - s*(ym + 1.0/ym) - nc)

    while true
        U = um * rand(Uniform())
        V = rand(Uniform())
        X = U/V
        if log(V) <= (t*log(X) - s*(X + 1/X) - nc)
            if lambda_old < 0.0 
                return alpha / X
            else
                return alpha * X
            end
        end
    end
end

function gig_concave(lambda, lambda_old, omega, alpha)
    xm = gig_mode(lambda,omega)
    x0 = omega/(1.0-lambda)
    k0 = exp((lambda - 1.0)*log(xm) - 0.5*omega*(xm + 1.0/xm))
    A = MVector{3,Float64}(undef)
    A[1] = k0 * x0
    if x0 >= 2.0/omega
        k1 = 0.0
        A[2] = 0.0
        k2 = x0^(lambda - 1.0)
        A[3] = k2 * 2.0 * exp(-omega*x0/2.0)/omega
    else
        k1 = exp(-omega)
        if lambda == 0.0
            A[2] = k1*log(2.0/(omega^2))
        else
            A[2] = k1 / lambda * ((2.0/omega)^lambda - x0^lambda)
        end
        k2 = (2.0/omega)^(lambda-1.0)
        A[3] = k2 * 2 * exp(-1.0)/omega
    end
    
    Atot = sum(A)
    while true
        V = Atot * rand(Uniform())
        hx = 0
        X = 0

        while true
            if V <= A[1]
                X = x0 * V / A[1]
                hx = k0
                break
            end

            V -= A[1]
            if V <= A[2]
                if lambda == 0 
                    X = omega * exp(exp(omega)*V)
                    hx = k1 / X
                else
                    X = (x0^lambda + (lambda / k1 * V))^(1.0/lambda)
                    hx = k1 * X^(lambda-1.0)
                end
                break
            end

            V -= A[2]
            if x0 > (2.0/omega)
                a = x0
            else
                a = 2.0/omega
            end
            X = -2.0/omega * log(exp(-omega/2.0 * a) - omega/(2.0*k2) * V)
            hx = k2 * exp(-omega/2.0 * X)
            break
        end
        U = rand(Uniform()) * hx
        if log(U) <= (lambda - 1.0) * log(X) - omega/2.0 * (X+1.0/X)
            if lambda_old < 0.0
                return alpha / X
            else
                return alpha * X
            end
        end
    end
end

function gig_mode(lambda, omega)
    if lambda >= 1.0
        return (sqrt((lambda - 1.0)^2 + omega^2) + lambda-1.0)/omega
    else
        return omega / (sqrt((1.0 - lambda)^2 + omega^2) + (1.0 - lambda))
    end
end
