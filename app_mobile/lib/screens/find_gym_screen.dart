import 'package:flutter/material.dart';
import 'package:geolocator/geolocator.dart';
import 'package:supabase_flutter/supabase_flutter.dart';
import '../services/checkin_service.dart';
import '../main.dart';

class FindGymScreen extends StatefulWidget {
  const FindGymScreen({super.key});

  @override
  State<FindGymScreen> createState() => _FindGymScreenState();
}

class _FindGymScreenState extends State<FindGymScreen> {
  List<Map<String, dynamic>> _gyms = [];
  bool _loading = true;
  String? _error;
  Position? _position;

  @override
  void initState() {
    super.initState();
    _loadNearbyGyms();
  }

  Future<void> _loadNearbyGyms() async {
    try {
      LocationPermission permission = await Geolocator.checkPermission();
      if (permission == LocationPermission.denied) {
        permission = await Geolocator.requestPermission();
      }
      if (permission == LocationPermission.denied ||
          permission == LocationPermission.deniedForever) {
        setState(() {
          _error = 'Location permission is required to find nearby gyms.';
          _loading = false;
        });
        return;
      }

      final position = await Geolocator.getCurrentPosition();
      _position = position;

      final results = await Supabase.instance.client.rpc('gyms_near', params: {
        'lat': position.latitude,
        'lng': position.longitude,
        'radius_km': 50,
      });

      setState(() {
        _gyms = List<Map<String, dynamic>>.from(results);
        _loading = false;
      });
    } catch (e) {
      setState(() {
        _error = 'Failed to load nearby gyms: $e';
        _loading = false;
      });
    }
  }

  double _distanceMiles(Map<String, dynamic> gym) {
    if (_position == null ||
        gym['latitude'] == null ||
        gym['longitude'] == null) {
      return 0;
    }
    final meters = Geolocator.distanceBetween(
      _position!.latitude,
      _position!.longitude,
      (gym['latitude'] as num).toDouble(),
      (gym['longitude'] as num).toDouble(),
    );
    return meters / 1609.344;
  }

  void _showGymDetail(Map<String, dynamic> gym) {
    final miles = _distanceMiles(gym);
    showModalBottomSheet(
      context: context,
      builder: (ctx) {
        return Padding(
          padding: const EdgeInsets.all(24),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Text(
                gym['name'] ?? 'Unknown Gym',
                style: const TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
              ),
              if (gym['address'] != null) ...[
                const SizedBox(height: 8),
                Text(gym['address']),
              ],
              const SizedBox(height: 8),
              Text('${miles.toStringAsFixed(1)} miles away'),
              const SizedBox(height: 16),
              ElevatedButton(
                onPressed: () async {
                  Navigator.pop(ctx);
                  await _checkIn(gym['id']);
                },
                child: const Text('Check In Here'),
              ),
            ],
          ),
        );
      },
    );
  }

  Future<void> _checkIn(String gymId) async {
    try {
      final profile = await supabaseService.fetchCurrentProfile();
      if (profile == null) {
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(content: Text('Could not load your profile.')),
          );
        }
        return;
      }

      await CheckinService.manualCheckIn(profile['id'], gymId);

      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Checked in!')),
        );
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Check-in failed: $e')),
        );
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Find a Gym')),
      body: _loading
          ? const Center(child: CircularProgressIndicator())
          : _error != null
              ? Center(
                  child: Padding(
                    padding: const EdgeInsets.all(24),
                    child: Text(_error!, textAlign: TextAlign.center),
                  ),
                )
              : _gyms.isEmpty
                  ? const Center(child: Text('No gyms found nearby.'))
                  : ListView.builder(
                      itemCount: _gyms.length,
                      itemBuilder: (context, index) {
                        final gym = _gyms[index];
                        final miles = _distanceMiles(gym);
                        return ListTile(
                          title: Text(gym['name'] ?? ''),
                          subtitle: Text('${miles.toStringAsFixed(1)} mi'),
                          trailing: const Icon(Icons.chevron_right),
                          onTap: () => _showGymDetail(gym),
                        );
                      },
                    ),
    );
  }
}
