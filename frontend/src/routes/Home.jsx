import { HeroSection, FeaturesSection, CTASection } from '../components/Home';

export const Home = () => {
    return (
        <div className="min-h-screen bg-[#1b1b1b] text-white">
            <HeroSection />
            <FeaturesSection />
            <CTASection />
        </div>
    );
};