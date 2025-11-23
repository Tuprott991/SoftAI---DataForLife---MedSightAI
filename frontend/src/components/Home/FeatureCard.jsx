export const FeatureCard = ({ icon: Icon, title, description, isVisible = false }) => {
    return (
        <div
            className={`group bg-white/5 backdrop-blur-sm border border-white/10 rounded-xl p-8 hover:bg-white/10 hover:border-teal-500/50 transition-all duration-500 hover:transform hover:scale-105 ${isVisible
                    ? 'opacity-100 translate-y-0'
                    : 'opacity-0 translate-y-8'
                }`}
        >
            <div className="w-14 h-14 bg-teal-500/20 rounded-lg flex items-center justify-center mb-6 group-hover:bg-teal-500/30 transition-colors">
                <Icon className="w-7 h-7 text-teal-500" />
            </div>
            <h3 className="text-xl font-bold mb-3">{title}</h3>
            <p className="text-gray-400">
                {description}
            </p>
        </div>
    );
};
