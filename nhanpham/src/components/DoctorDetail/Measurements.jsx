const getStatusColor = (status) => {
    switch (status) {
        case 'critical':
            return 'bg-red-500/10 text-red-400 border-red-500/20';
        case 'warning':
            return 'bg-yellow-500/10 text-yellow-400 border-yellow-500/20';
        case 'normal':
            return 'bg-teal-500/10 text-teal-400 border-teal-500/20';
        default:
            return 'bg-gray-500/10 text-gray-400 border-gray-500/20';
    }
};

export const Measurements = ({ metrics }) => {
    return (
        <div className="bg-[#1a1a1a] border border-white/10 rounded-xl overflow-hidden">
            <div className="px-4 py-3 border-b border-white/10 bg-[#141414]">
                <h3 className="text-sm font-semibold text-white">Measurements</h3>
            </div>
            <div className="p-3 grid grid-cols-2 gap-2">
                {metrics.map((metric, index) => (
                    <div key={index} className={`p-2.5 rounded-lg border ${getStatusColor(metric.status)}`}>
                        <p className="text-xs text-gray-500 mb-1">{metric.label}</p>
                        <p className="text-sm font-semibold">{metric.value}</p>
                    </div>
                ))}
            </div>
        </div>
    );
};
