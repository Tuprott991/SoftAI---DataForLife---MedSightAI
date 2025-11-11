import { useParams, Link } from 'react-router-dom';
import { ArrowLeft, User, Calendar, Activity, Droplet, MapPin, Phone, Mail, FileText } from 'lucide-react';
import { patientsData } from '../constants/patients';
import { Chatbot } from '../components/custom/Chatbot';

export const DoctorDetail = () => {
    const { id } = useParams();
    const patient = patientsData.find(p => p.id === parseInt(id));

    if (!patient) {
        return (
            <div className="min-h-screen bg-[#1b1b1b] text-white">
                <div className="container mx-auto px-6 py-8">
                    <Link
                        to="/doctor"
                        className="inline-flex items-center gap-2 text-gray-400 hover:text-white transition-colors mb-6"
                    >
                        <ArrowLeft className="w-5 h-5" />
                        <span>Back to Patient Records</span>
                    </Link>
                    <div className="text-center py-20">
                        <h1 className="text-3xl font-bold mb-4">Patient Not Found</h1>
                        <p className="text-gray-400">The patient record you're looking for doesn't exist.</p>
                    </div>
                </div>
            </div>
        );
    }

    const getStatusColor = (status) => {
        switch (status) {
            case 'Critical':
                return 'bg-red-500/20 text-red-400 border-red-500/30';
            case 'Under Treatment':
                return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30';
            case 'Stable':
                return 'bg-green-500/20 text-green-400 border-green-500/30';
            default:
                return 'bg-gray-500/20 text-gray-400 border-gray-500/30';
        }
    };

    // Mock medical images
    const medicalImages = [
        { id: 1, type: 'X-Ray', url: 'https://images.unsplash.com/photo-1530026405186-ed1f139313f8?w=400', date: '2025-11-10' },
        { id: 2, type: 'MRI Scan', url: 'https://images.unsplash.com/photo-1559757175-0eb30cd8c063?w=400', date: '2025-11-08' },
        { id: 3, type: 'CT Scan', url: 'https://images.unsplash.com/photo-1581594693702-fbdc51b2763b?w=400', date: '2025-11-05' },
        { id: 4, type: 'Ultrasound', url: 'https://images.unsplash.com/photo-1579154204601-01588f351e67?w=400', date: '2025-11-03' }
    ];

    return (
        <div className="min-h-screen bg-[#1b1b1b] text-white">
            <div className="container mx-auto px-6 py-8">
                {/* Back Button */}
                <Link
                    to="/doctor"
                    className="inline-flex items-center gap-2 text-gray-400 hover:text-white transition-colors mb-6"
                >
                    <ArrowLeft className="w-5 h-5" />
                    <span>Back to Patient Records</span>
                </Link>

                {/* Two Column Layout */}
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                    {/* Left Column - Patient Information */}
                    <div className="lg:col-span-2 space-y-6">
                        {/* Patient Header Card */}
                        <div className="bg-white/5 backdrop-blur-sm border border-white/10 rounded-xl p-6">
                            <div className="flex flex-col md:flex-row gap-6">
                                {/* Patient Photo */}
                                <div className="shrink-0">
                                    <div className="w-32 h-32 rounded-xl overflow-hidden border-2 border-teal-500/30">
                                        <img
                                            src={patient.image}
                                            alt={patient.name}
                                            className="w-full h-full object-cover"
                                        />
                                    </div>
                                </div>

                                {/* Patient Info */}
                                <div className="flex-1">
                                    <div className="flex items-start justify-between mb-3">
                                        <div>
                                            <h1 className="text-3xl font-bold mb-2">{patient.name}</h1>
                                            <p className="text-teal-400 text-lg">{patient.diagnosis}</p>
                                        </div>
                                        <span className={`px-4 py-2 rounded-lg text-sm font-semibold border ${getStatusColor(patient.status)}`}>
                                            {patient.status}
                                        </span>
                                    </div>

                                    <div className="grid grid-cols-2 md:grid-cols-3 gap-4 mt-6">
                                        <div className="flex items-center gap-2 text-gray-300">
                                            <User className="w-4 h-4 text-teal-500" />
                                            <div>
                                                <p className="text-xs text-gray-500">Age / Gender</p>
                                                <p className="font-medium">{patient.age} yrs / {patient.gender}</p>
                                            </div>
                                        </div>
                                        <div className="flex items-center gap-2 text-gray-300">
                                            <Droplet className="w-4 h-4 text-teal-500" />
                                            <div>
                                                <p className="text-xs text-gray-500">Blood Type</p>
                                                <p className="font-medium">{patient.bloodType}</p>
                                            </div>
                                        </div>
                                        <div className="flex items-center gap-2 text-gray-300">
                                            <Calendar className="w-4 h-4 text-teal-500" />
                                            <div>
                                                <p className="text-xs text-gray-500">Admitted</p>
                                                <p className="font-medium">{new Date(patient.admissionDate).toLocaleDateString()}</p>
                                            </div>
                                        </div>
                                        <div className="flex items-center gap-2 text-gray-300">
                                            <Activity className="w-4 h-4 text-teal-500" />
                                            <div>
                                                <p className="text-xs text-gray-500">Last Visit</p>
                                                <p className="font-medium">{new Date(patient.lastVisit).toLocaleDateString()}</p>
                                            </div>
                                        </div>
                                        <div className="flex items-center gap-2 text-gray-300">
                                            <Phone className="w-4 h-4 text-teal-500" />
                                            <div>
                                                <p className="text-xs text-gray-500">Contact</p>
                                                <p className="font-medium">+1 234 567 890</p>
                                            </div>
                                        </div>
                                        <div className="flex items-center gap-2 text-gray-300">
                                            <Mail className="w-4 h-4 text-teal-500" />
                                            <div>
                                                <p className="text-xs text-gray-500">Email</p>
                                                <p className="font-medium text-sm">patient@email.com</p>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>

                        {/* Medical History */}
                        <div className="bg-white/5 backdrop-blur-sm border border-white/10 rounded-xl p-6">
                            <h2 className="text-xl font-bold mb-4 flex items-center gap-2">
                                <FileText className="w-5 h-5 text-teal-500" />
                                Medical History
                            </h2>
                            <div className="space-y-3">
                                <div className="border-l-2 border-teal-500 pl-4 py-2">
                                    <p className="text-sm text-gray-400">Current Diagnosis</p>
                                    <p className="text-white font-medium">{patient.diagnosis}</p>
                                </div>
                                <div className="border-l-2 border-blue-500 pl-4 py-2">
                                    <p className="text-sm text-gray-400">Treatment Plan</p>
                                    <p className="text-white">Ongoing medication and regular monitoring</p>
                                </div>
                                <div className="border-l-2 border-purple-500 pl-4 py-2">
                                    <p className="text-sm text-gray-400">Allergies</p>
                                    <p className="text-white">Penicillin, Sulfa drugs</p>
                                </div>
                            </div>
                        </div>

                        {/* Medical Images */}
                        <div className="bg-white/5 backdrop-blur-sm border border-white/10 rounded-xl p-6">
                            <h2 className="text-xl font-bold mb-4">Medical Imaging Results</h2>
                            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                                {medicalImages.map((image) => (
                                    <div key={image.id} className="group relative rounded-lg overflow-hidden border border-white/10 hover:border-teal-500/50 transition-all cursor-pointer">
                                        <img
                                            src={image.url}
                                            alt={image.type}
                                            className="w-full h-32 object-cover group-hover:scale-110 transition-transform"
                                        />
                                        <div className="absolute inset-0 bg-linear-to-t from-black/80 to-transparent flex flex-col justify-end p-3">
                                            <p className="text-xs font-semibold text-white">{image.type}</p>
                                            <p className="text-xs text-gray-300">{new Date(image.date).toLocaleDateString()}</p>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>

                    {/* Right Column - Chatbot */}
                    <div className="lg:col-span-1">
                        <div className="sticky top-6">
                            <Chatbot patientName={patient.name} />
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default DoctorDetail;
