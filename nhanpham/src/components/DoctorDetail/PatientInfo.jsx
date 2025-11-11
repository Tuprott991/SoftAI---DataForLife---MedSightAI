export const PatientInfo = ({ patient }) => {
    return (
        <div className="relative inline-block group">
            <button className="bg-[#2a2a2a] text-white px-4 py-2 rounded-lg shadow-md hover:bg-[#383838] transition">
                Patient Info
            </button>

            <div className="absolute right-0 mt-2 w-64 bg-[#2a2a2a] p-4 rounded-lg shadow-lg shadow-black/40 opacity-0 group-hover:opacity-100 transition-opacity duration-300 pointer-events-none group-hover:pointer-events-auto z-10">
                <h3 className="text-lg font-bold mb-3">Personal Information</h3>
                <p className="text-sm text-gray-300">Name: {patient.name}</p>
                <p className="text-sm text-gray-300">Specialty: {patient.specialty}</p>
                <p className="text-sm text-gray-300">Experience: {patient.experience} years</p>
            </div>
        </div>
    );
};
