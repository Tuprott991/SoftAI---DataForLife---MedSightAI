import { useState, useEffect, useRef } from 'react';
import { Brain, Heart, Shield, GraduationCap, Activity, Stethoscope } from 'lucide-react';
import { FeatureCard } from './FeatureCard';

const features = [
    {
        icon: Brain,
        title: 'AI-Powered Diagnosis',
        description: 'Leverage advanced machine learning algorithms for accurate medical diagnoses and treatment recommendations.'
    },
    {
        icon: Heart,
        title: 'Patient Care Management',
        description: 'Comprehensive patient records, treatment history, and appointment scheduling in one unified platform.'
    },
    {
        icon: Shield,
        title: 'Secure & Compliant',
        description: 'Enterprise-grade security with full compliance to healthcare data protection standards and regulations.'
    },
    {
        icon: GraduationCap,
        title: 'Medical Education',
        description: 'Interactive learning modules, virtual simulations, and comprehensive medical resources for students.'
    },
    {
        icon: Activity,
        title: 'Real-time Analytics',
        description: 'Monitor patient vitals, track treatment progress, and analyze health trends with real-time data visualization.'
    },
    {
        icon: Stethoscope,
        title: 'Expert Collaboration',
        description: 'Connect with medical professionals worldwide, share insights, and collaborate on complex cases.'
    }
];

export const FeaturesSection = () => {
    const [visibleCards, setVisibleCards] = useState([]);
    const sectionRef = useRef(null);

    useEffect(() => {
        const observer = new IntersectionObserver(
            (entries) => {
                entries.forEach((entry) => {
                    if (entry.isIntersecting) {
                        features.forEach((_, index) => {
                            setTimeout(() => {
                                setVisibleCards((prev) => {
                                    if (!prev.includes(index)) {
                                        return [...prev, index];
                                    }
                                    return prev;
                                });
                            }, index * 150); // Delay 150ms giữa mỗi card
                        });
                    }
                });
            },
            {
                threshold: 0.1,
                rootMargin: '0px'
            }
        );

        if (sectionRef.current) {
            observer.observe(sectionRef.current);
        }

        return () => {
            if (sectionRef.current) {
                observer.unobserve(sectionRef.current);
            }
        };
    }, []);

    return (
        <div ref={sectionRef} className="container mx-auto px-6 py-20">
            <div className="text-center mb-16">
                <h2 className="text-3xl md:text-4xl font-bold mb-4">Why Choose MedSightAI?</h2>
                <p className="text-gray-400 text-lg">Advanced features designed for modern healthcare</p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
                {features.map((feature, index) => (
                    <FeatureCard
                        key={index}
                        icon={feature.icon}
                        title={feature.title}
                        description={feature.description}
                        isVisible={visibleCards.includes(index)}
                        delay={index * 150}
                    />
                ))}
            </div>
        </div>
    );
};
