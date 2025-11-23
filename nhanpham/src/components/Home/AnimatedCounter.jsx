import { useState, useEffect, useRef } from 'react';

export const AnimatedCounter = ({ end, duration = 2000, suffix = '', prefix = '', decimals = 0, icon: Icon, label }) => {
    const [count, setCount] = useState(0);
    const [isVisible, setIsVisible] = useState(false);
    const [hasAnimated, setHasAnimated] = useState(false);
    const counterRef = useRef(null);

    useEffect(() => {
        const observer = new IntersectionObserver(
            (entries) => {
                entries.forEach((entry) => {
                    if (entry.isIntersecting && !hasAnimated) {
                        setIsVisible(true);
                        setHasAnimated(true);
                    }
                });
            },
            {
                threshold: 0.3,
                rootMargin: '0px'
            }
        );

        if (counterRef.current) {
            observer.observe(counterRef.current);
        }

        return () => {
            if (counterRef.current) {
                observer.unobserve(counterRef.current);
            }
        };
    }, [hasAnimated]);

    useEffect(() => {
        if (!isVisible) return;

        let startTime;
        let animationFrame;

        const animate = (currentTime) => {
            if (!startTime) startTime = currentTime;
            const progress = Math.min((currentTime - startTime) / duration, 1);

            // Easing function for smooth animation (easeOutExpo)
            const easeOutExpo = progress === 1 ? 1 : 1 - Math.pow(2, -10 * progress);

            const currentCount = easeOutExpo * end;
            setCount(currentCount);

            if (progress < 1) {
                animationFrame = requestAnimationFrame(animate);
            } else {
                setCount(end);
            }
        };

        animationFrame = requestAnimationFrame(animate);

        return () => {
            if (animationFrame) {
                cancelAnimationFrame(animationFrame);
            }
        };
    }, [isVisible, end, duration]);

    const formatNumber = (num) => {
        if (decimals > 0) {
            return num.toFixed(decimals);
        }
        return Math.floor(num).toLocaleString();
    };

    return (
        <div
            ref={counterRef}
            className={`flex items-center gap-3 transition-all duration-700 ${isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4'
                }`}
        >
            {Icon && (
                <div className="w-12 h-12 bg-teal-500/20 rounded-lg flex items-center justify-center shrink-0">
                    <Icon className="w-6 h-6 text-teal-400" />
                </div>
            )}
            <div>
                <div className="flex items-baseline gap-1">
                    <span className="text-3xl md:text-4xl font-bold text-white tabular-nums">
                        {prefix}{formatNumber(count)}{suffix}
                    </span>
                </div>
                <span className="text-sm text-gray-400 font-medium">{label}</span>
            </div>
        </div>
    );
};
